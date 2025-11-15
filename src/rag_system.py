"""
Core RAG system for question answering (production version without evaluation).
"""

import logging
from typing import List, Tuple
import sys
from pathlib import Path

from langchain_classic.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_loader import load_documents
from src.chunking import split_text, save_chunks, load_saved_chunks
from src.vector_store import prepare_chunks_for_chroma, save_to_chroma, retrieve_top_k
from src.config import LLM_MODEL, LLM_TEMPERATURE
from src.prompts import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation system for board game manual Q&A.
    
    This is the production version without evaluation metrics.
    """
    
    def __init__(
        self,
        pdf_path: str = None,
        chunk_size: int = 300,
        chunk_overlap: int = 30,
        similarity_search: str = "cosine",
        embedding_model: str = "text-embedding-ada-002",
        llm_model: str = LLM_MODEL,
        llm_temperature: float = LLM_TEMPERATURE
    ):
        """
        Initialize RAG system.
        
        Args:
            pdf_path: Path to PDF document
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model
            llm_model: LLM model for answer generation
            llm_temperature: Temperature for LLM generation
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_search = similarity_search
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        
        self.db = None
        self.tmp_dir = None
        self.llm = None
        
        logger.info("Initializing RAG system...")
        self._setup()
    
    def _setup(self):
        """Load documents and create vector store."""
        # Load and process documents if file available
        if self.pdf_path is not None:
            logger.info(f"Loading PDF: {self.pdf_path}")
            docs = load_documents(self.pdf_path)
            
            # Chunk documents
            chunks = split_text(docs, self.chunk_size, self.chunk_overlap)
            chunks_path = Path(self.pdf_path).with_name("CATAN_chunks.json")
            save_chunks(chunks=chunks, path=chunks_path)
        else:
            # if file not available, use saved chunks of CATAN rulebook
            chunks = load_saved_chunks(path = "data/BoardGamesRuleBook/CATAN_chunks.json")

        # Prepare and store in vector DB
        chunks_for_chroma = prepare_chunks_for_chroma(chunks)
        self.db, self.tmp_dir = save_to_chroma(chunks_for_chroma, self.embedding_model, self.similarity_search)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=self.llm_model, temperature=self.llm_temperature)
        
        logger.info("RAG system initialized successfully")
    
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str, int, float]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            
        Returns:
            List of tuples: (source, content, chunk_id, relevance_score)
        """
        return retrieve_top_k(self.db, query, k=k)
    
    def generate_answer(
        self, 
        question: str, 
        context: List[str], 
        print_prompt: bool = False,
        prompt:str = "default"
    ) -> str:
        """
        Generate answer using LLM based on retrieved context.
        
        Args:
            question: User question
            context: List of relevant text chunks
            print_prompt: Whether to print the full prompt (for debugging)
            
        Returns:
            Generated answer as string
        """
        # Generate the prompt template with context and query
        context_text = "\n\n---\n\n".join(context)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATES[prompt])
        prompt = prompt_template.format(context=context_text, question=question)
        
        if print_prompt:
            print("=" * 60)
            print("PROMPT:")
            print("=" * 60)
            print(prompt)
            print("=" * 60)
        
        # Generate answer
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def answer_question(
        self, 
        question: str, 
        k: int = 3,
        print_prompt: bool = False,
        return_context: bool = False,
        prompt: str = "default"
    ) -> str:
        """
        End-to-end question answering: retrieve + generate.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            print_prompt: Whether to print the full prompt
            return_context: Whether to return context along with answer
            prompt: Selected prompt template
            
        Returns:
            Generated answer (or tuple of (answer, context) if return_context=True)
        """
        # Retrieve relevant chunks
        results = self.retrieve(question, k=k)
        context = [content for _, content, _, _ in results]
        
        # Generate answer
        answer = self.generate_answer(question, context, print_prompt=print_prompt, prompt=prompt)
        
        if return_context:
            return answer, context
        return answer
    
    def cleanup(self):
        """Clean up temporary resources."""
        if self.tmp_dir:
            import shutil
            try:
                shutil.rmtree(self.tmp_dir)
                logger.info(f"Cleaned up temporary directory: {self.tmp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")