"""
Run comprehensive evaluation of the RAG system.

Usage:
    python run_evaluation.py
"""

import shutil
from itertools import product

import pandas as pd

from evaluation import evaluate_retrieval, evaluate_generation
from config import (
    PDF_PATH,
    TRAINING_QA_PATH,
    CHUNK_SIZES,
    CHUNK_OVERLAPS,
    TOP_K_VALUES,
    EMBEDDING_MODEL,
    LLM_MODEL
)


def main():
    """Run evaluation with parameter grid search."""
    print("\n" + "=" * 70)
    print("   RAG System Evaluation")
    print("=" * 70 + "\n")
    
    retrieval_results = []
    generation_results = []
    
    # Grid search over parameters
    total_configs = len(CHUNK_SIZES) * len(CHUNK_OVERLAPS) * len(TOP_K_VALUES)
    print(f"Testing {total_configs} parameter configurations...\n")
    
    for idx, (chunk_size, chunk_overlap, k) in enumerate(
        product(CHUNK_SIZES, CHUNK_OVERLAPS, TOP_K_VALUES), 1
    ):
        print(f"[{idx}/{total_configs}] Configuration:")
        print(f"  • Chunk size: {chunk_size}")
        print(f"  • Chunk overlap: {chunk_overlap}")
        print(f"  • Top-k: {k}")
        print()
        
        try:
            # Evaluate retrieval
            print("  Running retrieval evaluation...")
            retrieval_result = evaluate_retrieval(
                pdf_path=str(PDF_PATH),
                training_qa_path=str(TRAINING_QA_PATH),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                k=k,
                embedding_model=EMBEDDING_MODEL
            )
            
            # Store retrieval results
            retrieval_results.append({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": k,
                "avg_dcg": retrieval_result["avg_dcg"],
                "avg_ndcg": retrieval_result["avg_ndcg"],
                "num_queries": retrieval_result["num_queries"]
            })
            
            # Evaluate generation
            print("  Running generation evaluation...")
            generation_result = evaluate_generation(
                query_results=retrieval_result["query_results"],
                llm_model=LLM_MODEL
            )
            
            # Store generation results
            generation_results.append({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": k,
                **generation_result
            })
            
            # Cleanup temporary directory
            if retrieval_result.get("tmp_dir"):
                try:
                    shutil.rmtree(retrieval_result["tmp_dir"])
                except Exception as e:
                    print(f"  ⚠️  Warning: Failed to cleanup temp dir: {e}")
            
            print(f"  ✅ Configuration {idx} completed\n")
            
        except Exception as e:
            print(f"  ❌ Error in configuration {idx}: {e}\n")
            continue
    
    # Save results
    print("=" * 70)
    print("Saving evaluation results...")
    print("=" * 70 + "\n")
    
    # Retrieval results
    retrieval_df = pd.DataFrame(retrieval_results)
    retrieval_df.to_csv("rag_retrieval_eval.csv", index=False)
    print("✅ Retrieval results saved to: rag_retrieval_eval.csv")
    print("\nTop 3 configurations by nDCG:")
    print(retrieval_df.nlargest(3, "avg_ndcg")[["chunk_size", "chunk_overlap", "top_k", "avg_ndcg"]])
    
    # Generation results
    generation_df = pd.DataFrame(generation_results)
    generation_df.to_csv("rag_generation_eval.csv", index=False)
    print("\n✅ Generation results saved to: rag_generation_eval.csv")
    print("\nTop 3 configurations by answer correctness:")
    print(generation_df.nlargest(3, "answer_correctness_mean")[
        ["chunk_size", "chunk_overlap", "top_k", "answer_correctness_mean"]
    ])
    
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()