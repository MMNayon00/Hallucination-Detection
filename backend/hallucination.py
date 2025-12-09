"""
Hallucination Detection and Scoring Module
Implements claim-level verification against retrieved evidence
"""

import numpy as np
from typing import List, Dict, Tuple
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Detects hallucinations by comparing generated answers against retrieved evidence.
    
    Methodology:
    1. Extract claims from generated answer (sentence-level segmentation)
    2. Compute semantic similarity between each claim and retrieved passages
    3. Score claims based on maximum similarity to any evidence
    4. Aggregate to overall hallucination score
    """
    
    def __init__(self, embedding_model, threshold: float = 0.45):
        """
        Initialize hallucination detector.
        
        Args:
            embedding_model: Sentence embedding model
            threshold: Hallucination threshold (higher = more permissive)
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract individual claims from text (sentence segmentation).
        
        Args:
            text: Answer text to segment
            
        Returns:
            List of claim sentences
        """
        # Simple sentence splitting using regex
        # Handles periods, exclamation marks, question marks
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Filter out very short sentences (likely artifacts)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return claims if claims else [text]  # Return full text if no sentences found
    
    def compute_claim_scores(
        self, 
        claims: List[str], 
        evidence: List[Dict[str, any]]
    ) -> List[float]:
        """
        Compute verification score for each claim against evidence.
        
        Args:
            claims: List of claim sentences
            evidence: List of evidence dictionaries from retrieval
            
        Returns:
            List of scores (0-1) for each claim
        """
        if not evidence:
            logger.warning("No evidence available for verification")
            return [0.0] * len(claims)
        
        # Extract evidence snippets
        evidence_texts = [e["snippet"] for e in evidence]
        
        # Encode claims and evidence
        claim_embeddings = self.embedding_model.encode(
            claims,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        
        evidence_embeddings = self.embedding_model.encode(
            evidence_texts,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        
        claim_embeddings = np.array(claim_embeddings)
        evidence_embeddings = np.array(evidence_embeddings)
        
        # Compute cosine similarity matrix (claims x evidence)
        # Normalize embeddings
        claim_norms = np.linalg.norm(claim_embeddings, axis=1, keepdims=True)
        evidence_norms = np.linalg.norm(evidence_embeddings, axis=1, keepdims=True)
        
        claim_embeddings_norm = claim_embeddings / (claim_norms + 1e-8)
        evidence_embeddings_norm = evidence_embeddings / (evidence_norms + 1e-8)
        
        # Similarity matrix
        similarity_matrix = np.dot(claim_embeddings_norm, evidence_embeddings_norm.T)
        
        # For each claim, take maximum similarity to any evidence
        claim_scores = np.max(similarity_matrix, axis=1)
        
        return claim_scores.tolist()
    
    def detect_hallucination(
        self,
        answer: str,
        evidence: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Detect if answer contains hallucinations.
        
        Args:
            answer: Generated answer text
            evidence: Retrieved evidence passages
            
        Returns:
            Dictionary with detection results:
            {
                'is_hallucinated': bool,
                'hallucination_score': float,
                'claim_scores': List[float],
                'claims': List[str],
                'low_confidence_claims': List[str]
            }
        """
        # Extract claims
        claims = self.extract_claims(answer)
        
        # Compute claim scores
        claim_scores = self.compute_claim_scores(claims, evidence)
        
        # Aggregate to hallucination score
        # Hallucination score = 1 - mean(claim_scores)
        # Higher score = more likely hallucinated
        mean_claim_score = np.mean(claim_scores) if claim_scores else 0.0
        hallucination_score = 1.0 - mean_claim_score
        
        # Determine if hallucinated
        is_hallucinated = hallucination_score > self.threshold
        
        # Identify low-confidence claims (below a threshold)
        low_confidence_threshold = 0.5
        low_confidence_claims = [
            claims[i] for i, score in enumerate(claim_scores)
            if score < low_confidence_threshold
        ]
        
        result = {
            "is_hallucinated": bool(is_hallucinated),
            "hallucination_score": float(hallucination_score),
            "claim_scores": claim_scores,
            "claims": claims,
            "low_confidence_claims": low_confidence_claims,
            "mean_evidence_support": float(mean_claim_score)
        }
        
        logger.info(
            f"Hallucination detection: score={hallucination_score:.3f}, "
            f"hallucinated={is_hallucinated}, claims={len(claims)}"
        )
        
        return result
    
    def set_threshold(self, threshold: float):
        """
        Update hallucination detection threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        logger.info(f"Hallucination threshold updated to {threshold}")
