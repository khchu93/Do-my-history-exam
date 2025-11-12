"""
Vector store operations using Chroma and OpenAI embeddings.
"""

import tempfile
import logging
from typing import List, Tuple
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


def prepare_chunks_for_chroma(chunks: List[Document]) -> List[Document]:
    """
    Filter complex metadata for Chroma compatibility.
    
    Why needed?
    - Chroma only supports simple types (str, int, float, bool)
    - Complex types (lists, dicts) cause serialization errors
    - We keep complex metadata in separate 'relevant_chunks' list
    
    Args:
        chunks: List of Document chunks
        
    Returns:
        Documents with filtered metadata safe for Chroma
        
    Raises:
        VectorStoreError: If metadata filtering fails
    """
    try:
        retrievable_docs = []
        
        for chunk_idx, chunk in enumerate(chunks):
            # Filter to simple metadata types
            filtered_doc = filter_complex_metadata([chunk])[0]
            
            # Add chunk_id for later lookup
            filtered_doc.metadata["chunk_id"] = chunk_idx
            
            retrievable_docs.append(filtered_doc)
        
        logger.info(f"Prepared {len(retrievable_docs)} chunks for Chroma")
        return retrievable_docs
        
    except Exception as e:
        logger.error(f"Metadata filtering failed: {str(e)}")
        raise VectorStoreError(f"Failed to prepare chunks: {str(e)}") from e


def save_to_chroma(chunks: List[Document], embedding_model: str = "text-embedding-ada-002") -> Tuple[Chroma, str]:
    """
    Create and persist Chroma vector store.
    
    Note: This creates a temporary database!
    - Ensures fresh embeddings
    - Avoids stale data issues
    - For production, consider incremental updates
    
    Args:
        chunks: List of Document chunks (with simple metadata)
        embedding_model: OpenAI embedding model name
        
    Returns:
        Tuple of (Chroma vector store, temporary directory path)
        
    Raises:
        VectorStoreError: If vector store creation fails
    """
    try:
        # Create an isolated temporary directory
        tmp_dir = tempfile.mkdtemp(prefix="chroma_eval_")

        # Initialize embedding model
        embeddings = OpenAIEmbeddings(model=embedding_model)

        # Create the Chroma vector store from documents
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=tmp_dir
        )

        logger.info(f"Temporary Chroma DB created at: {tmp_dir}")
        return db, tmp_dir
        
    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
        raise VectorStoreError(f"Failed to create vector store: {str(e)}") from e


def retrieve_top_k(
    db: Chroma, 
    query: str, 
    k: int = 3
) -> List[Tuple[str, str, int, float]]:
    """
    Retrieve top-k most similar chunks for a query.
    
    Args:
        db: Chroma vector store
        query: Query string
        k: Number of documents to retrieve
    
    Returns:
        List of tuples: (source, content, chunk_id, relevance_score)
        
    Raises:
        VectorStoreError: If retrieval fails
    """
    try:
        if k <= 0:
            raise VectorStoreError(f"k must be positive, got {k}")
        
        logger.debug(f"Retrieving top-{k} chunks for query: {query[:50]}...")
        
        results = db.similarity_search_with_relevance_scores(query, k=k)
        
        formatted_results = [
            (
                doc.metadata.get("source", "unknown"),
                doc.page_content,
                doc.metadata.get("chunk_id", -1),
                score
            )
            for doc, score in results
        ]
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        raise VectorStoreError(f"Failed to retrieve documents: {str(e)}") from e