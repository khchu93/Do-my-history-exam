"""
Document loading and preprocessing utilities.
"""

import re
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from exceptions import DocumentLoadError

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text to handle encoding inconsistencies between PDF and JSON.
    
    This is critical because:
    - PDFs may have curly quotes/apostrophes: "", '', '
    - JSON files typically use straight quotes: ", '
    - Mismatches break pattern matching for ground truth annotation
    
    Args:
        text: Raw text string
        
    Returns:
        Normalized text with standardized quotes and collapsed whitespace
    """
    # Convert curly quotes to straight quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    
    # Collapse all whitespace (newlines, tabs, multiple spaces) to single space
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def load_documents(pdf_path: str) -> List[Document]:
    """
    Load PDF and clean text content.
    
    Why cleaning matters:
    - PDFs often have inconsistent spacing/newlines
    - Normalized text improves embedding quality
    - Standardized format makes pattern matching reliable
    
    Args:
        pdf_path: Path to the board game manual PDF
        
    Returns:
        List of Document objects (one per page) with cleaned text
        
    Raises:
        DocumentLoadError: If PDF cannot be loaded or is empty
    """
    try:
        # Validate file exists
        if not Path(pdf_path).exists():
            raise DocumentLoadError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        page_docs = loader.load()
        
        if not page_docs:
            raise DocumentLoadError(f"No content extracted from PDF: {pdf_path}")
        
        logger.info(f"Loaded {len(page_docs)} pages from PDF")
        
        # Clean text and filter metadata
        for page_doc in page_docs:
            # Normalize whitespace
            clean_text = normalize_text(page_doc.page_content)
            page_doc.page_content = clean_text
            
            # Keep only essential metadata to avoid Chroma serialization issues
            allowed_keys = {"source", "page"}
            page_doc.metadata = {
                k: v for k, v in page_doc.metadata.items() 
                if k in allowed_keys
            }
        
        return page_docs
        
    except Exception as e:
        if isinstance(e, DocumentLoadError):
            raise
        logger.error(f"Unexpected error loading PDF: {str(e)}")
        raise DocumentLoadError(f"Failed to load PDF: {str(e)}") from e