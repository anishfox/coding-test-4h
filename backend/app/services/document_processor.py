"""
Document processing service using Docling

Implements:
1. Parse PDF documents using Docling
2. Extract text, images, and tables
3. Store extracted content in database
4. Generate embeddings for text chunks
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
from app.core.config import settings
import os
import json
import uuid
import time
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process PDF documents and extract multimodal content.
    
    Handles:
    - PDF parsing with Docling
    - Text extraction and chunking
    - Image extraction and storage
    - Table extraction and storage
    - Embedding generation for chunks
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document using Docling.
        
        Implementation steps:
        1. Update document status to 'processing'
        2. Use Docling to parse the PDF
        3. Extract and save text chunks
        4. Extract and save images
        5. Extract and save tables
        6. Generate embeddings for text chunks
        7. Update document status to 'completed'
        8. Handle errors appropriately
        
        Args:
            file_path: Path to the uploaded PDF file
            document_id: Database ID of the document
            
        Returns:
            {
                "status": "success" or "error",
                "text_chunks": <count>,
                "images": <count>,
                "tables": <count>,
                "processing_time": <seconds>
            }
        """
        start_time = time.time()
        text_chunks_count = 0
        images_count = 0
        tables_count = 0
        
        try:
            # Update status to processing
            await self._update_document_status(document_id, "processing")
            logger.info(f"Starting document processing for document_id: {document_id}")
            
            # Parse PDF using Docling
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                doc_result = converter.convert(file_path)
                logger.info(f"PDF parsed successfully with Docling. Total pages: {len(doc_result.pages)}")
            except Exception as e:
                logger.warning(f"Docling conversion failed, falling back to PyPDF: {e}")
                # Fallback to PyPDF for simple text extraction
                import pypdf
                
                class SimpleDocResult:
                    def __init__(self, pages):
                        self.pages = pages
                
                class SimplePage:
                    def __init__(self, page_num, text):
                        self.text = text
                
                pdf_pages = []
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text()
                        pdf_pages.append(SimplePage(page_num, text))
                
                doc_result = SimpleDocResult(pdf_pages)
                logger.info(f"PDF parsed with PyPDF fallback. Total pages: {len(pdf_pages)}")
            
            logger.info(f"PDF parsed successfully. Total pages: {len(doc_result.pages)}")
            
            # Extract text and create chunks
            all_text_chunks = []
            for page_num, page in enumerate(doc_result.pages, 1):
                if page.text:
                    chunks = self._chunk_text(
                        text=page.text,
                        document_id=document_id,
                        page_number=page_num
                    )
                    all_text_chunks.extend(chunks)
            
            text_chunks_count = len(all_text_chunks)
            logger.info(f"Extracted {text_chunks_count} text chunks")
            
            # Save text chunks with embeddings
            await self._save_text_chunks(all_text_chunks, document_id)
            
            # Extract and save images
            if hasattr(doc_result, 'images') and doc_result.images:
                for img_idx, image in enumerate(doc_result.images, 1):
                    try:
                        # Get image data and metadata
                        image_data = image.data if hasattr(image, 'data') else None
                        if image_data:
                            await self._save_image(
                                image_data=image_data,
                                document_id=document_id,
                                page_number=getattr(image, 'page_number', 0),
                                metadata={
                                    "caption": getattr(image, 'caption', f"Figure {img_idx}"),
                                    "index": img_idx
                                }
                            )
                            images_count += 1
                    except Exception as e:
                        logger.warning(f"Error saving image {img_idx}: {e}")
                        continue
            
            # Extract and save tables
            if hasattr(doc_result, 'tables') and doc_result.tables:
                for table_idx, table in enumerate(doc_result.tables, 1):
                    try:
                        await self._save_table(
                            table_data=table,
                            document_id=document_id,
                            page_number=getattr(table, 'page_number', 0),
                            metadata={
                                "caption": getattr(table, 'caption', f"Table {table_idx}"),
                                "index": table_idx
                            }
                        )
                        tables_count += 1
                    except Exception as e:
                        logger.warning(f"Error saving table {table_idx}: {e}")
                        continue
            
            # Update document with extraction results
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.total_pages = len(doc_result.pages)
                document.text_chunks_count = text_chunks_count
                document.images_count = images_count
                document.tables_count = tables_count
                self.db.commit()
            
            # Update status to completed
            processing_time = time.time() - start_time
            await self._update_document_status(document_id, "completed")
            
            logger.info(
                f"Document {document_id} processing completed in {processing_time:.2f}s. "
                f"Extracted: {text_chunks_count} chunks, {images_count} images, {tables_count} tables"
            )
            
            return {
                "status": "success",
                "text_chunks": text_chunks_count,
                "images": images_count,
                "tables": tables_count,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)
            processing_time = time.time() - start_time
            await self._update_document_status(
                document_id, 
                "error",
                error_message=f"Processing failed: {str(e)}"
            )
            
            return {
                "status": "error",
                "text_chunks": text_chunks_count,
                "images": images_count,
                "tables": tables_count,
                "processing_time": processing_time,
                "error": str(e)
            }
    
    def _chunk_text(self, text: str, document_id: int, page_number: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        Implements text chunking strategy:
        - Splits by sentences/paragraphs
        - Maintains context with overlap
        - Keeps metadata (page number, position, etc.)
        
        Args:
            text: Text content from page
            document_id: Document ID
            page_number: Page number in document
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        
        if not text or not text.strip():
            return chunks
        
        # Split by paragraphs first (double newlines)
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(paragraph) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "document_id": document_id,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "metadata": {
                            "source": "text",
                            "paragraph_count": current_chunk.count('\n') + 1
                        }
                    })
                    chunk_index += 1
                    
                    # Create overlap by keeping last part of previous chunk
                    words = current_chunk.split()
                    overlap_words = int(len(words) * (self.chunk_overlap / self.chunk_size))
                    current_chunk = " ".join(words[-overlap_words:]) if overlap_words > 0 else ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "document_id": document_id,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "metadata": {
                    "source": "text",
                    "paragraph_count": current_chunk.count('\n') + 1
                }
            })
        
        logger.debug(f"Created {len(chunks)} chunks from page {page_number}")
        return chunks
    
    async def _save_text_chunks(self, chunks: List[Dict[str, Any]], document_id: int):
        """
        Save text chunks to database with embeddings.
        
        Implementation:
        - Generate embeddings for each chunk
        - Store in database with pgvector
        - Link related images/tables in metadata
        
        Args:
            chunks: List of chunk dictionaries
            document_id: Document ID
        """
        logger.info(f"Saving {len(chunks)} text chunks for document {document_id}")
        
        for chunk in chunks:
            try:
                # Generate embedding for chunk
                embedding = await self.vector_store.generate_embedding(chunk["content"])
                
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for chunk {chunk['chunk_index']}")
                    continue
                
                # Create DocumentChunk record
                doc_chunk = DocumentChunk(
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    chunk_metadata=json.dumps(chunk.get("metadata", {}))
                )
                
                self.db.add(doc_chunk)
                
            except Exception as e:
                logger.error(
                    f"Error saving chunk {chunk['chunk_index']} "
                    f"(page {chunk['page_number']}): {e}"
                )
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully saved text chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Error committing chunks to database: {e}")
            self.db.rollback()
            raise
    
    async def _save_image(
        self, 
        image_data: bytes, 
        document_id: int, 
        page_number: int,
        metadata: Dict[str, Any]
    ) -> Optional[DocumentImage]:
        """
        Save an extracted image.
        
        Implementation:
        - Save image file to disk
        - Create DocumentImage record
        - Extract caption if available
        
        Args:
            image_data: Image binary data
            document_id: Document ID
            page_number: Page number where image appears
            metadata: Image metadata (caption, index, etc.)
            
        Returns:
            Created DocumentImage record or None if failed
        """
        try:
            # Create images directory if it doesn't exist
            images_dir = os.path.join(settings.UPLOAD_DIR, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Generate unique filename
            image_id = str(uuid.uuid4())
            image_filename = f"doc_{document_id}_page_{page_number}_img_{image_id}.png"
            image_path = os.path.join(images_dir, image_filename)
            
            # Save image to disk
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Create database record
            doc_image = DocumentImage(
                document_id=document_id,
                file_path=f"images/{image_filename}",  # Relative path for serving
                page_number=page_number,
                caption=metadata.get("caption", f"Figure {metadata.get('index', 1)}"),
                width=metadata.get("width"),
                height=metadata.get("height"),
                image_metadata=json.dumps({
                    "source": "docling",
                    "index": metadata.get("index", 1)
                })
            )
            
            self.db.add(doc_image)
            self.db.commit()
            
            logger.info(f"Saved image to {image_path}")
            return doc_image
            
        except Exception as e:
            logger.error(f"Error saving image for document {document_id}: {e}")
            self.db.rollback()
            return None
    
    async def _save_table(
        self,
        table_data: Any,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> Optional[DocumentTable]:
        """
        Save an extracted table.
        
        Implementation:
        - Render table as image
        - Store structured data as JSON
        - Create DocumentTable record
        - Extract caption if available
        
        Args:
            table_data: Table object from Docling
            document_id: Document ID
            page_number: Page number where table appears
            metadata: Table metadata (caption, index, etc.)
            
        Returns:
            Created DocumentTable record or None if failed
        """
        try:
            # Create tables directory if it doesn't exist
            tables_dir = os.path.join(settings.UPLOAD_DIR, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            
            # Generate unique filename for table image
            table_id = str(uuid.uuid4())
            table_filename = f"doc_{document_id}_page_{page_number}_tbl_{table_id}.png"
            table_path = os.path.join(tables_dir, table_filename)
            
            # Extract table data as structured format
            table_json = {}
            try:
                # Try to extract table structure if available
                if hasattr(table_data, 'to_dict'):
                    table_json = table_data.to_dict()
                elif hasattr(table_data, 'data'):
                    table_json = table_data.data
                else:
                    table_json = {"raw": str(table_data)}
            except Exception as e:
                logger.warning(f"Could not extract table structure: {e}")
                table_json = {"raw": str(table_data)}
            
            # Try to render table as image
            try:
                # If Docling provides image data
                if hasattr(table_data, 'data') and isinstance(table_data.data, bytes):
                    with open(table_path, 'wb') as f:
                        f.write(table_data.data)
                else:
                    # Fallback: create a simple placeholder image
                    # In production, use libraries like PIL to render the table
                    logger.warning(f"Table image not available, creating placeholder for table {metadata.get('index')}")
                    # Create a simple text-based representation
                    table_json['note'] = "Table image rendering not available"
                    
            except Exception as e:
                logger.warning(f"Could not save table image: {e}")
            
            # Create database record
            doc_table = DocumentTable(
                document_id=document_id,
                image_path=f"tables/{table_filename}",  # Relative path for serving
                data=json.dumps(table_json),
                page_number=page_number,
                caption=metadata.get("caption", f"Table {metadata.get('index', 1)}"),
                table_metadata=json.dumps({
                    "source": "docling",
                    "index": metadata.get("index", 1)
                })
            )
            
            self.db.add(doc_table)
            self.db.commit()
            
            logger.info(f"Saved table to database for document {document_id}, page {page_number}")
            return doc_table
            
        except Exception as e:
            logger.error(f"Error saving table for document {document_id}: {e}")
            self.db.rollback()
            return None
    
    async def _update_document_status(
        self, 
        document_id: int, 
        status: str, 
        error_message: str = None
    ):
        """
        Update document processing status.
        
        This is implemented as an example.
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = status
            if error_message:
                document.error_message = error_message
            self.db.commit()
