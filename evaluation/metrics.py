"""
Evaluation metrics for retrieval quality (DCG and nDCG).
"""

import logging
from typing import List

import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.exceptions import EvaluationError

logger = logging.getLogger(__name__)


def dcg(relevance_scores: List[float]) -> float:
    """
    Calculate Discounted Cumulative Gain.
    
    Formula: DCG = Σ(rel_i / log2(i + 2)) for i in range(len(scores))
    
    Why log2(i + 2)?
    - Position 0: log2(2) = 1 (no discount)
    - Position 1: log2(3) = 1.58 (small discount)
    - Position 2: log2(4) = 2 (larger discount)
    - Later positions are increasingly discounted
    
    Args:
        relevance_scores: List of relevance scores (coverage values)
        
    Returns:
        DCG score
        
    Raises:
        EvaluationError: If calculation fails
    """
    try:
        if not relevance_scores:
            return 0.0
        
        dcg_value = np.sum([
            rel / np.log2(idx + 2)
            for idx, rel in enumerate(relevance_scores)
        ])
        
        return float(dcg_value)
        
    except Exception as e:
        logger.error(f"DCG calculation failed: {str(e)}")
        raise EvaluationError(f"Failed to calculate DCG: {str(e)}") from e


def ndcg_at_k(relevance_scores: List[float]) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    nDCG = DCG / IDCG
    
    Why normalize?
    - Makes scores comparable across queries
    - Range: [0, 1] where 1 = perfect ranking
    - Accounts for different numbers of relevant items
    
    Args:
        relevance_scores: List of relevance scores
        
    Returns:
        nDCG score between 0 and 1
        
    Raises:
        EvaluationError: If calculation fails
    """
    try:
        if not relevance_scores:
            return 0.0
        
        # Calculate DCG with actual ranking
        dcg_value = dcg(relevance_scores)
        
        # Calculate ideal DCG (perfect ranking)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg_value = dcg(ideal_scores)
        
        # Avoid division by zero
        if idcg_value == 0:
            logger.warning("IDCG is 0, returning nDCG=0")
            return 0.0
        
        ndcg_value = dcg_value / idcg_value
        return float(ndcg_value)
        
    except Exception as e:
        logger.error(f"nDCG calculation failed: {str(e)}")
        raise EvaluationError(f"Failed to calculate nDCG: {str(e)}") from e