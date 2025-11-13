"""
Run comprehensive evaluation of the RAG system.

Usage:
    python run_evaluation.py
"""

import shutil
import gc
import time
from itertools import product

import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.evaluation import evaluate_retrieval, evaluate_generation
from src.config import (
    PDF_PATH,
    TRAINING_QA_PATH,
    CHUNK_SIZES,
    CHUNK_OVERLAPS,
    TOP_K_VALUES,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PROMPT_TEMPLATE
)

def safe_close_chroma(db):
    """
    Safely close Chroma database connection and release file locks.
    """
    if db is None:
        return
    
    try:
        # Delete the database object to release locks
        del db
        gc.collect()
        time.sleep(0.2)  # Give OS time to release file handles
    except Exception as e:
        print(f"    ⚠️ Error closing Chroma DB: {e}")

def safe_empty_dir(tmp_dir, retries=3, delay=0.3):
    """
    Safely empties all contents inside tmp_dir without removing the directory itself.
    Handles locked files on Windows gracefully.
    """
    if not tmp_dir or not Path(tmp_dir).exists():
        return

    tmp_path = Path(tmp_dir)

    for attempt in range(retries):
        try:
            # Iterate and delete individual items
            for item in tmp_path.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                except PermissionError:
                    # Locked file: skip and try again after a delay
                    time.sleep(delay)
                    continue
            return
        except Exception as e:
            print(f"    ⚠️ Cleanup attempt {attempt+1} failed: {e}")
            time.sleep(delay)
            gc.collect()

    print(f"    ⚠️ Gave up cleaning some locked files in {tmp_dir}")

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
        # Skip invalid combinations where overlap >= chunk_size
        if chunk_overlap >= chunk_size:
            print(f"⏭️  Skipping invalid config: chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size})")
            continue

        print(f"[{idx}/{total_configs}] Configuration:")
        print(f"  • Chunk size: {chunk_size}")
        print(f"  • Chunk overlap: {chunk_overlap}")
        print(f"  • Top-k: {k}")
        print()
        
        db = None
        tmp_dir = Path(f"evaluation/tmp/chroma_{chunk_size}_{chunk_overlap}_{k}")

        # 🧹 STEP 1: Make sure directory is clean BEFORE starting
        if tmp_dir.exists():
            print(f"  ⚙️  Removing old Chroma directory: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        
        try:
            # 🧩 STEP 2: Run retrieval evaluation
            print("  Running retrieval evaluation...")
            retrieval_result = evaluate_retrieval(
                pdf_path=str(PDF_PATH),
                training_qa_path=str(TRAINING_QA_PATH),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                k=k,
                embedding_model=EMBEDDING_MODEL,
                persist_dir=str(tmp_dir)
            )
            
            # ⚠️ CRITICAL: Extract db for proper cleanup
            db = retrieval_result.get("db")
            tmp_dir = retrieval_result.get("tmp_dir", tmp_dir)
            
            # Store retrieval results (without db object)
            retrieval_results.append({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": k,
                "avg_dcg": retrieval_result["avg_dcg"],
                "avg_ndcg": retrieval_result["avg_ndcg"],
                "mrr": retrieval_result["mrr"],
                "mmrs": retrieval_result["mmrs"],
                "mhitrate_k": retrieval_result["mhitrate_k"],
                "num_queries": retrieval_result["num_queries"]
            })
            
            # # Evaluate generation
            # print("  Running generation evaluation...")
            # generation_result = evaluate_generation(
            #     query_results=retrieval_result["query_results"],
            #     llm_model=LLM_MODEL,
            #     prompt=PROMPT_TEMPLATE
            # )
            
            # # Store generation results
            # generation_results.append({
            #     "chunk_size": chunk_size,
            #     "chunk_overlap": chunk_overlap,
            #     "top_k": k,
            #     **generation_result
            # })

            print(f"  ✅ Configuration {idx} completed")
            
        except Exception as e:
            print(f"  ❌ Error in configuration {idx}: {e}")
            # Continue to next configuration instead of crashing
            
        finally:
            # ✅ CRITICAL: Close DB BEFORE cleanup
            print("  🧹 Closing database connection...")
            safe_close_chroma(db)
            
            print("  🧹 Cleaning up resources...")
            safe_empty_dir(tmp_dir)
            print()
            
            # Optional: Force garbage collection every 10 iterations
            if idx % 10 == 0:
                print(f"  🔄 Forcing garbage collection (iteration {idx})...")
                gc.collect()
                time.sleep(0.3)
    
    # Save results
    print("=" * 70)
    print("Saving evaluation results...")
    print("=" * 70 + "\n")
    
    # Retrieval results
    if retrieval_results:
        retrieval_df = pd.DataFrame(retrieval_results)
        retrieval_df.to_csv("evaluation/results_csv/rag_retrieval_eval.csv", index=False)
        print("✅ Retrieval results saved to: evaluation/results_csv/rag_retrieval_eval.csv")
        print(f"\n📊 Completed {len(retrieval_results)}/{total_configs} configurations")
        print("\nTop 3 configurations by nDCG:")
        print(retrieval_df.nlargest(3, "avg_ndcg")[["chunk_size", "chunk_overlap", "top_k", "avg_ndcg"]])
    else:
        print("❌ No results to save (all configurations failed)")
    
    # # Generation results
    # if generation_results:
    #     generation_df = pd.DataFrame(generation_results)
    #     generation_df.to_csv("evaluation/results_csv/rag_generation_eval.csv", index=False)
    #     print("\n✅ Generation results saved to: evaluation/results_csv/rag_generation_eval.csv")
    #     print("\nTop 3 configurations by answer correctness:")
    #     print(generation_df.nlargest(3, "answer_correctness_mean")[
    #         ["chunk_size", "chunk_overlap", "top_k", "answer_correctness_mean"]
    #     ])

    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
