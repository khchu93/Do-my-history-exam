"""
Document chunking and coverage calculation utilities.
"""

import copy
import logging
from typing import List
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.exceptions import ChunkingError, EvaluationError

logger = logging.getLogger(__name__)


def split_text(docs: List[Document], chunk_size: int = 300, chunk_overlap: int = 30) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Why chunk?
    - Embeddings work better on focused, semantic units
    - Smaller chunks = more precise retrieval
    - Overlap ensures we don't split important context
    
    Why these defaults?
    - chunk_size=300: ~75 tokens, good for rule-specific content
    - chunk_overlap=30: 10% overlap preserves context at boundaries
    
    Args:
        docs: List of Document objects
        chunk_size: Target size for each chunk (characters)
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunk Documents with start_index in metadata
        
    Raises:
        ChunkingError: If chunking process fails
    """
    try:
        if chunk_size <= 0:
            raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")
        
        if chunk_overlap < 0:
            raise ChunkingError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        logger.info(f"Splitting documents with chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # Use character count
            add_start_index=True  # Critical: needed for coverage calculation
        )
        
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
        
    except Exception as e:
        if isinstance(e, ChunkingError):
            raise
        logger.error(f"Chunking failed: {str(e)}")
        raise ChunkingError(f"Failed to split documents: {str(e)}") from e


def compute_overlap(span_start: int, span_end: int, chunk_start: int, chunk_end: int) -> int:
    """
    Compute character overlap between a relevance span and a chunk.
    
    Example:
        Span:  [10, 30]  (relevant text from annotation)
        Chunk: [20, 50]  (text chunk)
        Overlap: [20, 30] = 10 characters
    
    Args:
        span_start: Start index of relevance span
        span_end: End index of relevance span (exclusive)
        chunk_start: Start index of chunk
        chunk_end: End index of chunk (exclusive)
        
    Returns:
        Number of overlapping characters
    """
    overlap_start = max(span_start, chunk_start)
    overlap_end = min(span_end, chunk_end)
    return max(0, overlap_end - overlap_start)


def generate_relevant_chunks_with_coverage(chunks: List[Document]) -> List[Document]:
    """
    Calculate coverage scores for chunks containing ground truth spans.
    
    Coverage = (overlap_length / relevance_span_length)
    
    Why coverage?
    - Measures "how much of the relevant content is in this chunk"
    - Coverage=1.0: entire relevant span is in the chunk
    - Coverage=0.5: only half the relevant content is present
    - Coverage=0.0: chunk doesn't contain relevant content
    
    This is better than binary relevance because:
    - Distinguishes between partial and complete matches
    - Handles cases where spans cross chunk boundaries
    - Provides granular relevance scores for nDCG calculation
    
    Args:
        chunks: List of Document chunks
        
    Returns:
        List of Documents containing only relevant chunks with coverage scores
        
    Raises:
        EvaluationError: If coverage calculation fails
    """
    try:
        relevant_chunks = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = chunk.metadata.get("start_index", 0)
            chunk_end = chunk_start + len(chunk.page_content)
            relevance_spans = chunk.metadata.get("relevance_spans", [])
            
            # Skip chunks without any ground truth annotations
            if not relevance_spans:
                continue
            
            # Create copy to avoid modifying original
            annotated_chunk = copy.deepcopy(chunk)
            annotated_chunk.metadata["coverage_per_query"] = []
            
            for span in relevance_spans:
                qa_id = span["qa_id"]
                
                # Calculate how much of the span overlaps with this chunk
                overlap_len = compute_overlap(
                    span["start"], span["end"], 
                    chunk_start, chunk_end
                )
                
                relevance_len = span["end"] - span["start"]
                
                # Avoid division by zero
                if relevance_len == 0:
                    logger.warning(f"Zero-length relevance span for qa_id={qa_id}")
                    continue
                
                coverage = overlap_len / relevance_len
                
                # Skip queries with no overlap
                if coverage == 0:
                    continue
                
                annotated_chunk.metadata["coverage_per_query"].append({
                    "qa_id": qa_id,
                    "coverage": coverage
                })
            
            # Only keep chunks that have at least one relevant query
            if annotated_chunk.metadata["coverage_per_query"]:
                annotated_chunk.metadata["chunk_id"] = chunk_idx
                relevant_chunks.append(annotated_chunk)
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks out of {len(chunks)} total")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Coverage calculation failed: {str(e)}")
        raise EvaluationError(f"Failed to calculate coverage: {str(e)}") from e


def get_coverage(chunk_id: int, qa_id: str, relevant_chunks: List[Document]) -> float:
    """
    Retrieve coverage score for a specific chunk and query.
    
    This is a lookup function that connects:
    - Retrieved chunk (by chunk_id from vector search)
    - Query (by qa_id from evaluation set)
    - Ground truth coverage (pre-computed in relevant_chunks)
    
    Args:
        chunk_id: ID of the retrieved chunk
        qa_id: ID of the query being evaluated
        relevant_chunks: List of annotated chunks with coverage scores
        
    Returns:
        Coverage score (0-1), or 0 if not found
    """
    for chunk in relevant_chunks:
        if chunk.metadata.get("chunk_id") != chunk_id:
            continue
        
        for coverage_entry in chunk.metadata.get("coverage_per_query", []):
            if coverage_entry["qa_id"] == qa_id:
                return coverage_entry["coverage"]
    
    # Return 0 if chunk has no coverage for this query
    return 0.0