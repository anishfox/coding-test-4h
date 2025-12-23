"""
Vector store service using pgvector.

Implements:
1. Generate embeddings for text chunks using sentence-transformers
2. Store embeddings in PostgreSQL with pgvector
3. Perform similarity search using cosine distance
4. Link related images and tables
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text, and_
from app.models.document import DocumentChunk, DocumentImage, DocumentTable
from app.core.config import settings
import logging
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for document embeddings and similarity search.
    
    Handles:
    - Embedding generation with sentence-transformers
    - Vector storage with pgvector
    - Similarity search using cosine distance
    - Related content retrieval (images and tables)
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.embeddings_model = None
        self._init_embeddings()
        self._ensure_extension()
    
    def _init_embeddings(self):
        """
        Initialize embedding model.
        
        Uses sentence-transformers for local embeddings.
        No API keys required - runs locally.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence-transformers model initialized: all-MiniLM-L6-v2")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled.
        
        This is implemented as an example.
        """
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception as e:
            print(f"pgvector extension already exists or error: {e}")
            self.db.rollback()
    
    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text.
        
        Uses HuggingFace sentence-transformers (no API key required).
        Returns 384-dimensional vectors using all-MiniLM-L6-v2 model.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of shape (384,) or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            if self.embeddings_model is None:
                logger.error("Embeddings model not initialized")
                return None
            
            # Generate embedding
            embedding = self.embeddings_model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def store_chunk(
        self, 
        content: str, 
        document_id: int,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[DocumentChunk]:
        """
        Store a text chunk with its embedding.
        
        Implementation:
        1. Generate embedding for content
        2. Create DocumentChunk record
        3. Store in database with embedding
        4. Include metadata (related images, tables, etc.)
        
        Args:
            content: Text content
            document_id: Document ID
            page_number: Page number
            chunk_index: Index of chunk in document
            metadata: Additional metadata (related_images, related_tables, etc.)
            
        Returns:
            Created DocumentChunk or None if failed
        """
        try:
            # Generate embedding
            embedding = await self.generate_embedding(content)
            if embedding is None:
                logger.warning(f"Failed to generate embedding for chunk {chunk_index}")
                return None
            
            # Create DocumentChunk record
            doc_chunk = DocumentChunk(
                document_id=document_id,
                content=content,
                embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                page_number=page_number,
                chunk_index=chunk_index,
                metadata=json.dumps(metadata) if metadata else None
            )
            
            self.db.add(doc_chunk)
            self.db.commit()
            self.db.refresh(doc_chunk)
            
            logger.debug(f"Stored chunk {chunk_index} for document {document_id}")
            return doc_chunk
            
        except Exception as e:
            logger.error(f"Error storing chunk: {e}")
            self.db.rollback()
            return None
    
    async def similarity_search(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Implementation:
        1. Generate embedding for query
        2. Use pgvector's cosine similarity (<=> operator)
        3. Filter by document_id if provided
        4. Return top k results with scores
        5. Include related images and tables in results
        
        Args:
            query: Search query text
            document_id: Optional document ID to filter
            k: Number of results to return
            
        Returns:
            [
                {
                    "content": "...",
                    "score": 0.95,
                    "page_number": 3,
                    "metadata": {...},
                    "related_images": [...],
                    "related_tables": [...]
                }
            ]
        """
        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query)
            if query_embedding is None:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Convert to list format for pgvector
            query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
            
            # Build SQL query with pgvector similarity search
            sql = """
                SELECT 
                    id,
                    content,
                    page_number,
                    chunk_index,
                    metadata,
                    document_id,
                    1 - (embedding <=> :query_embedding) as similarity
                FROM document_chunks
            """
            
            # Add document filter if specified
            if document_id:
                sql += "\nWHERE document_id = :document_id"
            
            # Order by similarity and limit results
            sql += "\nORDER BY embedding <=> :query_embedding\nLIMIT :k"
            
            # Execute query
            query_obj = text(sql)
            query_obj = query_obj.bindparams(
                query_embedding=query_embedding_list,
                k=k
            )
            
            if document_id:
                query_obj = query_obj.bindparams(document_id=document_id)
            
            results = self.db.execute(query_obj).fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                chunk_data = {
                    "id": row.id,
                    "content": row.content,
                    "score": float(row.similarity),
                    "page_number": row.page_number,
                    "chunk_index": row.chunk_index,
                    "document_id": row.document_id,
                    "metadata": json.loads(row.chunk_metadata) if row.chunk_metadata else {},
                }
                
                # Get related images and tables for this chunk's page
                related_content = await self.get_related_content_by_page(
                    document_id=row.document_id,
                    page_number=row.page_number
                )
                
                chunk_data["related_images"] = related_content.get("images", [])
                chunk_data["related_tables"] = related_content.get("tables", [])
                
                formatted_results.append(chunk_data)
            
            logger.debug(f"Found {len(formatted_results)} similar chunks for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    async def get_related_content(
        self,
        chunk_ids: List[int]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get related images and tables for given chunks.
        
        Implementation:
        - Query DocumentImage and DocumentTable by page numbers
        - Return organized by type (images, tables)
        
        Args:
            chunk_ids: List of DocumentChunk IDs
            
        Returns:
            {
                "images": [{id, file_path, caption, page_number, url}, ...],
                "tables": [{id, image_path, caption, page_number, data, url}, ...]
            }
        """
        try:
            if not chunk_ids:
                return {"images": [], "tables": []}
            
            # Get page numbers from chunks
            chunks = self.db.query(DocumentChunk.page_number).filter(
                DocumentChunk.id.in_(chunk_ids)
            ).all()
            
            page_numbers = list(set([chunk.page_number for chunk in chunks if chunk.page_number]))
            
            if not page_numbers:
                return {"images": [], "tables": []}
            
            # Get document_id from first chunk
            first_chunk = self.db.query(DocumentChunk.document_id).filter(
                DocumentChunk.id == chunk_ids[0]
            ).first()
            
            if not first_chunk:
                return {"images": [], "tables": []}
            
            document_id = first_chunk.document_id
            
            # Query images and tables for these pages
            images = self.db.query(DocumentImage).filter(
                and_(
                    DocumentImage.document_id == document_id,
                    DocumentImage.page_number.in_(page_numbers)
                )
            ).all()
            
            tables = self.db.query(DocumentTable).filter(
                and_(
                    DocumentTable.document_id == document_id,
                    DocumentTable.page_number.in_(page_numbers)
                )
            ).all()
            
            # Format results
            images_data = [
                {
                    "id": img.id,
                    "file_path": img.file_path,
                    "caption": img.caption,
                    "page_number": img.page_number,
                    "url": f"/uploads/{img.file_path}"
                }
                for img in images
            ]
            
            tables_data = [
                {
                    "id": tbl.id,
                    "image_path": tbl.image_path,
                    "caption": tbl.caption,
                    "page_number": tbl.page_number,
                    "data": json.loads(tbl.data) if tbl.data else {},
                    "url": f"/uploads/{tbl.image_path}"
                }
                for tbl in tables
            ]
            
            return {"images": images_data, "tables": tables_data}
            
        except Exception as e:
            logger.error(f"Error retrieving related content: {e}")
            return {"images": [], "tables": []}
    
    async def get_related_content_by_page(
        self,
        document_id: int,
        page_number: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get images and tables for a specific page.
        
        Args:
            document_id: Document ID
            page_number: Page number
            
        Returns:
            {images: [...], tables: [...]}
        """
        try:
            # Query images and tables for this page
            images = self.db.query(DocumentImage).filter(
                and_(
                    DocumentImage.document_id == document_id,
                    DocumentImage.page_number == page_number
                )
            ).all()
            
            tables = self.db.query(DocumentTable).filter(
                and_(
                    DocumentTable.document_id == document_id,
                    DocumentTable.page_number == page_number
                )
            ).all()
            
            # Format results
            images_data = [
                {
                    "id": img.id,
                    "file_path": img.file_path,
                    "caption": img.caption,
                    "page_number": img.page_number,
                    "url": f"/uploads/{img.file_path}"
                }
                for img in images
            ]
            
            tables_data = [
                {
                    "id": tbl.id,
                    "image_path": tbl.image_path,
                    "caption": tbl.caption,
                    "page_number": tbl.page_number,
                    "data": json.loads(tbl.data) if tbl.data else {},
                    "url": f"/uploads/{tbl.image_path}"
                }
                for tbl in tables
            ]
            
            return {"images": images_data, "tables": tables_data}
            
        except Exception as e:
            logger.error(f"Error retrieving page content: {e}")
            return {"images": [], "tables": []}
