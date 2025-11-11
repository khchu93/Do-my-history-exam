"""
Ground truth Q&A annotation integration using Aho-Corasick pattern matching.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import ahocorasick
from langchain_core.documents import Document

from exceptions import AnnotationError
from document_loader import normalize_text

logger = logging.getLogger(__name__)


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load JSON file containing training Q&A pairs.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        AnnotationError: If file cannot be loaded or parsed
    """
    try:
        if not Path(json_path).exists():
            raise AnnotationError(f"JSON file not found: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON from: {json_path}")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        raise AnnotationError(f"Failed to parse JSON: {str(e)}") from e
    except Exception as e:
        if isinstance(e, AnnotationError):
            raise
        logger.error(f"Unexpected error loading JSON: {str(e)}")
        raise AnnotationError(f"Failed to load JSON: {str(e)}") from e


def load_training_qa_to_docs(training_qas_path: str, page_docs: List[Document]) -> List[Document]:
    """
    Annotate documents with ground truth relevance spans using Aho-Corasick.
    
    Why Aho-Corasick?
    - Efficient multi-pattern matching: O(n + m + z) vs O(n*m) for naive search
    - n = document length, m = total pattern length, z = matches
    - Critical when searching 100+ patterns across large documents
    
    Process:
    1. Build automaton with all relevant chunks from training Q&A
    2. Scan each page once to find all matching spans
    3. Store span metadata (qa_id, page, start/end indices)
    
    Args:
        training_qas_path: Path to JSON with training Q&A pairs
        page_docs: List of Document objects from PDF
        
    Returns:
        Documents annotated with relevance_spans in metadata
        
    Raises:
        AnnotationError: If annotation process fails
    """
    try:
        training_data = load_json(training_qas_path)
        training_qas = training_data.get("training_qas", [])
        
        if not training_qas:
            logger.warning("No training Q&As found in JSON")
            return page_docs
        
        logger.info(f"Processing {len(training_qas)} training Q&A pairs")
        
        # Build Aho-Corasick automaton for efficient pattern matching
        automaton = ahocorasick.Automaton()
        
        for qa_idx, qa in enumerate(training_qas):
            qa["relevance_spans"] = []  # Initialize spans list
            
            for chunk_text in qa.get("relevant_chunks", []):
                chunk_text_normalized = normalize_text(chunk_text)
                
                # Store tuple: (qa_index, original_chunk_text)
                # qa_index allows us to map back to the question
                automaton.add_word(chunk_text_normalized, (qa_idx, chunk_text_normalized))
        
        automaton.make_automaton()  # Compile the automaton
        logger.info("Aho-Corasick automaton built successfully")
        
        # Search all pages for relevant spans
        total_spans = 0
        for page_doc in page_docs:
            page_text = normalize_text(page_doc.page_content)
            page_num = page_doc.metadata.get("page")
            page_doc.metadata["relevance_spans"] = []
            
            # Iterate through all matches in this page
            for end_idx, (qa_idx, chunk_text) in automaton.iter(page_text):
                start_idx = end_idx - len(chunk_text) + 1  # +1 because end_idx is inclusive
                
                span = {
                    "qa_id": training_qas[qa_idx]["id"],
                    "page": page_num,
                    "start": start_idx,
                    "end": end_idx + 1  # Make end exclusive for easier indexing
                }
                page_doc.metadata["relevance_spans"].append(span)
                total_spans += 1
        
        logger.info(f"Found {total_spans} relevance spans across all pages")
        return page_docs
        
    except Exception as e:
        if isinstance(e, AnnotationError):
            raise
        logger.error(f"Annotation failed: {str(e)}")
        raise AnnotationError(f"Failed to annotate documents: {str(e)}") from e