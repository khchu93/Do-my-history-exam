"""
Complete evaluation pipeline with ground truth annotations and metrics.
"""

import logging
from typing import Dict, Any, List

import sys
from pathlib import Path
import numpy as np
from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_loader import load_documents
from src.annotation import load_training_qa_to_docs, load_json
from src.chunking import split_text, generate_relevant_chunks_with_coverage, get_coverage
from src.vector_store import prepare_chunks_for_chroma, save_to_chroma, retrieve_top_k
from evaluation.metrics import dcg, ndcg_at_k, reciprocal_rank, mean_retrieval_similarity, hit_rate_at_k
from src.exceptions import RAGEvaluationError, EvaluationError
from src.config import LLM_MODEL
from src.prompts import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


def generate_answer(question: str, context: List[str], llm, prompt: str) -> str:
    """
    Generate answer using LLM (for evaluation purposes).
    
    Args:
        question: User question
        context: List of relevant text chunks
        llm: LLM instance
        prompt: selected prompt template
        
    Returns:
        Generated answer as string
    """
    from langchain_classic.prompts import ChatPromptTemplate
    
    context_text = "\n\n---\n\n".join(context)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["default"])
    prompt = prompt_template.format(context=context_text, question=question)
    
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


def evaluate_retrieval(
    pdf_path: str,
    training_qa_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 30,
    k: int = 3,
    embedding_model: str = "text-embedding-ada-002",
    persist_dir: str = None
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality using DCG and nDCG metrics.
    
    Pipeline:
    1. Load and clean PDF
    2. Annotate with ground truth Q&A
    3. Chunk documents
    4. Calculate coverage for relevant chunks
    5. Create vector store
    6. For each query: retrieve top-k and calculate metrics
    7. Report average DCG and nDCG
    
    Args:
        pdf_path: Path to board game manual PDF
        training_qa_path: Path to training Q&A JSON
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        k: Number of documents to retrieve
        embedding_model: OpenAI embedding model
        persist_dir: Directory to persist Chroma DB (optional)
        
    Returns:
        Dictionary with evaluation results (including db and tmp_dir for cleanup)
        
    Raises:
        RAGEvaluationError: If any pipeline stage fails
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Retrieval Evaluation Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load PDF
        docs = load_documents(pdf_path)
        
        # Step 2: Annotate with ground truth
        docs_with_qa = load_training_qa_to_docs(training_qa_path, docs)
        
        # Step 3: Chunk documents
        chunks = split_text(docs_with_qa, chunk_size, chunk_overlap)
        
        # Step 4: Calculate coverage for relevant chunks
        relevant_chunks = generate_relevant_chunks_with_coverage(chunks)
        
        # Step 5: Prepare and store in vector DB
        chunks_for_chroma = prepare_chunks_for_chroma(chunks)
        db, tmp_dir = save_to_chroma(chunks_for_chroma, embedding_model, persist_dir)
        
        # Step 6: Load evaluation queries
        qa_data = load_json(training_qa_path)
        evaluation_qas = qa_data.get("training_qas", [])
        
        if not evaluation_qas:
            raise EvaluationError("No evaluation queries found in JSON")
        
        logger.info(f"Evaluating on {len(evaluation_qas)} queries")
        
        # Step 7: Evaluate each query
        dcg_values = []
        ndcg_values = []
        rr_values = []
        mrs_values = []
        hitrate_k_values = []
        query_results = []
        
        for qa in evaluation_qas:
            qa_id = qa.get("id")
            question = qa.get("question")
            gt_answer = qa.get("answer")
            
            if not question:
                logger.warning(f"Skipping query with missing question: {qa_id}")
                continue
            
            # Retrieve top-k chunks
            top_k_results = retrieve_top_k(db, question, k=k)
            top_k = []
            
            # Calculate coverage scores for retrieved chunks
            coverage_scores = []
            similarity_scores = []
            for source, content, chunk_id, similarity_score in top_k_results:
                coverage = get_coverage(chunk_id, qa_id, relevant_chunks)
                coverage_scores.append(coverage)
                similarity_scores.append(similarity_score)
                top_k.append(content)
            
            # Calculate metrics
            query_dcg = dcg(coverage_scores)
            query_ndcg = ndcg_at_k(coverage_scores)
            query_rr = reciprocal_rank(coverage_scores)
            query_mrs = mean_retrieval_similarity(similarity_scores)
            query_hitrate = hit_rate_at_k(coverage_scores)

            dcg_values.append(query_dcg)
            ndcg_values.append(query_ndcg)
            rr_values.append(query_rr)
            mrs_values.append(query_mrs)
            hitrate_k_values.append(query_hitrate)

            query_results.append({
                "qa_id": qa_id,
                "question": question,
                "top_k_content": top_k,
                "gt_answer": gt_answer,
                "coverage_scores": coverage_scores,
                "dcg": query_dcg,
                "ndcg": query_ndcg,
                "rr": rr_values,
                "mrs": mrs_values,
                "hitrate_k": hitrate_k_values
            })
        
        # Calculate averages
        avg_dcg = float(np.mean(dcg_values))
        avg_ndcg = float(np.mean(ndcg_values))
        mrr = float(np.mean(rr_values))
        mmrs = float(np.mean(mrs_values))
        mhitrate_k = float(np.mean(hitrate_k_values))
        
        logger.info("=" * 60)
        logger.info(f"Average DCG:  {avg_dcg:.4f}")
        logger.info(f"Average nDCG: {avg_ndcg:.4f}")
        logger.info("=" * 60)
        
        return {
            "avg_dcg": avg_dcg,
            "avg_ndcg": avg_ndcg,
            "mrr": mrr,
            "mmrs": mmrs,
            "mhitrate_k": mhitrate_k,
            "num_queries": len(evaluation_qas),
            "k": k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "query_results": query_results,
            "db": db,  # ⚠️ CRITICAL: Return db for proper cleanup
            "tmp_dir": tmp_dir
        }
        
    except Exception as e:
        if isinstance(e, RAGEvaluationError):
            raise
        logger.error(f"Pipeline failed: {str(e)}")
        raise RAGEvaluationError(f"Evaluation pipeline failed: {str(e)}") from e


def evaluate_generation(
    query_results: List[Dict[str, Any]],
    llm_model: str = LLM_MODEL,
    prompt: str = "default"
) -> Dict[str, Any]:
    """
    Evaluate answer generation quality using RAGAS metrics.
    
    Args:
        query_results: Results from retrieval evaluation
        llm_model: LLM model for generation
        prompt: selected prompt template
        
    Returns:
        Dictionary with generation evaluation metrics
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Generation Evaluation")
        logger.info("=" * 60)
        
        # Initialize LLM
        llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Generate answers for each query
        evaluation_rows = []
        for query_result in query_results:
            question = query_result.get("question")
            top_k_content = query_result.get("top_k_content")
            gt_answer = query_result.get("gt_answer")
            
            answer = generate_answer(question, top_k_content, llm, prompt)
            
            evaluation_rows.append({
                "question": question,
                "contexts": top_k_content,
                "answer": answer,
                "reference": gt_answer,
            })
        
        # Create RAGAS dataset
        ragas_eval_dataset = Dataset.from_list(evaluation_rows)
        
        # Run evaluation
        scores = evaluate(
            ragas_eval_dataset,
            metrics=[
                answer_correctness,
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
            ],
            llm=llm,
        )
        
        # Calculate statistics
        results = {
            "answer_correctness_mean": float(np.mean(scores["answer_correctness"])),
            "answer_correctness_std": float(np.std(scores["answer_correctness"])),
            "answer_relevancy_mean": float(np.mean(scores["answer_relevancy"])),
            "answer_relevancy_std": float(np.std(scores["answer_relevancy"])),
            "faithfulness_mean": float(np.mean(scores["faithfulness"])),
            "faithfulness_std": float(np.std(scores["faithfulness"])),
            "context_precision_mean": float(np.mean(scores["context_precision"])),
            "context_precision_std": float(np.std(scores["context_precision"])),
            "context_recall_mean": float(np.mean(scores["context_recall"])),
            "context_recall_std": float(np.std(scores["context_recall"])),
        }
        
        logger.info("=" * 60)
        logger.info("Generation Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Generation evaluation failed: {str(e)}")
        raise EvaluationError(f"Failed to evaluate generation: {str(e)}") from e