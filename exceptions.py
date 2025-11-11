"""
Custom exception classes for the RAG evaluation system.
"""


class RAGEvaluationError(Exception):
    """Base exception for RAG evaluation system"""
    pass


class DocumentLoadError(RAGEvaluationError):
    """Raised when document loading fails"""
    pass


class AnnotationError(RAGEvaluationError):
    """Raised when Q&A annotation processing fails"""
    pass


class ChunkingError(RAGEvaluationError):
    """Raised when document chunking fails"""
    pass


class VectorStoreError(RAGEvaluationError):
    """Raised when vector store operations fail"""
    pass


class EvaluationError(RAGEvaluationError):
    """Raised when metric calculation fails"""
    pass