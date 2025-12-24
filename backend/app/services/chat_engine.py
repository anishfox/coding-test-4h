"""
Chat engine service for multimodal RAG.

Implements:
1. Process user messages with RAG pipeline
2. Search for relevant context using vector store
3. Find related images and tables
4. Generate responses using Gemini LLM
5. Support multi-turn conversations
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.conversation import Conversation, Message
from app.services.vector_store import VectorStore
from app.core.config import settings
import time
import logging
import json

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Multimodal chat engine with RAG.
    
    Handles:
    - Conversation management with multi-turn support
    - Vector-based semantic search for context retrieval
    - Gemini LLM integration for response generation
    - Multimodal output (text + images + tables)
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """
        Initialize Gemini LLM.
        """
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.llm = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info(f"Gemini LLM initialized: {settings.GEMINI_MODEL}")
        except ImportError:
            logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            raise
    
    
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate multimodal response.
        
        Implementation steps:
        1. Load conversation history (for multi-turn support)
        2. Search vector store for relevant context
        3. Find related images and tables
        4. Build prompt with context and history
        5. Generate response using LLM
        6. Format response with sources (text, images, tables)
        7. Save message and response to database
        
        Args:
            conversation_id: Conversation ID
            message: User message
            document_id: Optional document ID to scope search
            
        Returns:
            {
                "answer": "...",
                "sources": [
                    {"type": "text", "content": "...", "page": 3, "score": 0.95},
                    {"type": "image", "url": "/uploads/...", "caption": "..."},
                    {"type": "table", "url": "/uploads/...", "data": {...}}
                ],
                "processing_time": 2.5
            }
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing message for conversation {conversation_id}")
            
            # 1. Load conversation history
            history = await self._load_conversation_history(
                conversation_id=conversation_id,
                limit=6  # Load last 6 messages for context
            )
            
            # 2. Search for relevant context
            context = await self._search_context(
                query=message,
                document_id=document_id,
                k=5  # Top 5 chunks
            )
            
            # 3. Find related images and tables
            media = await self._find_related_media(context)
            
            # 4. Generate response using LLM
            answer = await self._generate_response(
                message=message,
                context=context,
                history=history,
                media=media
            )
            
            # 5. Format sources
            sources = self._format_sources(context, media)
            
            processing_time = time.time() - start_time
            logger.info(f"Message processed in {processing_time:.2f}s")
            
            return {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            processing_time = time.time() - start_time
            return {
                "answer": f"Error processing your message: {str(e)}",
                "sources": [],
                "processing_time": processing_time
            }
    
    async def _load_conversation_history(
        self,
        conversation_id: int,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Load recent conversation history.
        
        Implementation:
        - Load last N messages from conversation
        - Format for LLM context
        - Include both user and assistant messages
        
        Args:
            conversation_id: Conversation ID
            limit: Number of recent messages to load
            
        Returns:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        try:
            # Query recent messages ordered by creation time
            messages = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(desc(Message.created_at)).limit(limit).all()
            
            # Reverse to get chronological order
            messages = list(reversed(messages))
            
            # Format for LLM
            history = []
            for msg in messages:
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            logger.debug(f"Loaded {len(history)} messages for conversation {conversation_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            return []
    
    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using vector store.
        
        Implementation:
        - Use vector store similarity search
        - Filter by document if specified
        - Return relevant chunks with metadata
        
        Args:
            query: User query text
            document_id: Optional document ID to filter
            k: Number of results to return
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Use vector store similarity search
            results = await self.vector_store.similarity_search(
                query=query,
                document_id=document_id,
                k=k
            )
            
            logger.debug(f"Found {len(results)} relevant chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching context: {e}")
            return []
    
    async def _find_related_media(
        self,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related images and tables from context chunks.
        
        Implementation:
        - Extract chunk IDs from results
        - Use vector store to get related content
        - Return with URLs for frontend display
        
        Args:
            context_chunks: List of search result chunks
            
        Returns:
            {
                "images": [...],
                "tables": [...]
            }
        """
        try:
            if not context_chunks:
                return {"images": [], "tables": []}
            
            # Extract chunk IDs
            chunk_ids = [chunk["id"] for chunk in context_chunks if "id" in chunk]
            
            if not chunk_ids:
                return {"images": [], "tables": []}
            
            # Use vector store to get related content
            related_content = await self.vector_store.get_related_content(chunk_ids)
            
            logger.debug(
                f"Found {len(related_content.get('images', []))} images and "
                f"{len(related_content.get('tables', []))} tables"
            )
            
            return related_content
            
        except Exception as e:
            logger.error(f"Error finding related media: {e}")
            return {"images": [], "tables": []}
    
    async def _generate_response(
        self,
        message: str,
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate response using Gemini LLM.
        
        Implementation:
        - Build comprehensive prompt with context
        - Include conversation history for context
        - Mention available images and tables
        - Call Gemini API
        - Return generated answer
        
        Args:
            message: User message
            context: Retrieved context chunks
            history: Conversation history
            media: Related images and tables
            
        Returns:
            Generated response text
        """
        try:
            # Build context string from chunks
            context_text = ""
            for i, chunk in enumerate(context[:5], 1):  # Top 5 chunks
                score = chunk.get("score", 0)
                content = chunk.get("content", "")
                page = chunk.get("page_number", "?")
                context_text += f"\n[Source {i}, Page {page}, Relevance: {score:.2%}]\n{content}\n"
            
            # Build media references
            media_text = ""
            if media.get("images"):
                media_text += "\n\nAvailable Figures:\n"
                for img in media["images"][:3]:
                    media_text += f"- {img.get('caption', 'Figure')} (Page {img.get('page_number')}): {img.get('url')}\n"
            
            if media.get("tables"):
                media_text += "\nAvailable Tables:\n"
                for tbl in media["tables"][:3]:
                    media_text += f"- {tbl.get('caption', 'Table')} (Page {tbl.get('page_number')}): {tbl.get('url')}\n"
            
            # Build conversation history context
            history_text = ""
            for msg in history[-4:]:  # Last 4 messages
                role = "You" if msg["role"] == "user" else "Assistant"
                history_text += f"\n{role}: {msg['content']}"
            
            # Build the prompt
            system_prompt = """You are a helpful assistant that answers questions about documents.

You have access to:
1. Document content (extracted text chunks with relevance scores)
2. Figures and diagrams from the document
3. Tables and data from the document

When answering:
- Use the provided context to answer accurately
- Reference figures and tables when relevant
- Cite the specific pages where information comes from
- Be concise but informative
- If information is not in the context, say so clearly"""
            
            user_prompt = f"""Previous conversation context:{history_text}

Relevant document context:{context_text}{media_text}

Question: {message}

Please provide a comprehensive answer based on the context above."""
            
            # Call Gemini API
            import google.generativeai as genai
            
            response = self.llm.generate_content(
                [system_prompt, user_prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=1024
                )
            )
            
            answer = response.text if response.text else "Unable to generate response"
            logger.debug(f"Generated response with {len(answer)} characters")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error generating a response: {str(e)}"
    
    def _format_sources(
        self,
        context: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Format sources for response.
        
        Combines text chunks, images, and tables into structured source list.
        
        Args:
            context: Retrieved text chunks with scores
            media: Images and tables by page
            
        Returns:
            Formatted list of sources for frontend display
        """
        sources = []
        
        # Add text sources (top 3)
        for chunk in context[:3]:
            content = chunk.get("content", "")
            truncated_content = content[:200] + "..." if len(content) > 200 else content
            sources.append({
                "type": "text",
                "content": truncated_content,
                "page_number": chunk.get("page_number"),
                "score": round(chunk.get("score", 0.0), 3),
                "chunk_index": chunk.get("chunk_index")
            })
        
        # Add image sources
        for image in media.get("images", []):
            sources.append({
                "type": "image",
                "url": image.get("url"),
                "caption": image.get("caption", "Figure"),
                "page_number": image.get("page_number"),
                "id": image.get("id")
            })
        
        # Add table sources
        for table in media.get("tables", []):
            sources.append({
                "type": "table",
                "url": table.get("url"),
                "caption": table.get("caption", "Table"),
                "page_number": table.get("page_number"),
                "data": table.get("data", {}),
                "id": table.get("id")
            })
        
        return sources
