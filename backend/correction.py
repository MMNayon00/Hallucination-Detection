"""
Answer Correction Module
Generates corrected answers when hallucinations are detected
"""

from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerCorrector:
    """
    Corrects hallucinated answers using evidence-grounded prompt engineering.
    """
    
    def __init__(self, model_manager):
        """
        Initialize answer corrector.
        
        Args:
            model_manager: ModelManager instance with LLM
        """
        self.model_manager = model_manager
    
    def correct_answer(
        self,
        question: str,
        original_answer: str,
        evidence: List[Dict[str, any]],
        low_confidence_claims: List[str] = None
    ) -> str:
        """
        Generate a corrected answer grounded in retrieved evidence.
        
        Args:
            question: Original question
            original_answer: Initial (potentially hallucinated) answer
            evidence: Retrieved evidence passages
            low_confidence_claims: Claims identified as low confidence
            
        Returns:
            Corrected answer string
        """
        if not evidence:
            logger.warning("No evidence available for correction")
            return original_answer
        
        # Construct evidence context
        evidence_context = self._format_evidence(evidence)
        
        # Create correction prompt
        prompt = self._create_correction_prompt(
            question,
            evidence_context,
            low_confidence_claims
        )
        
        # Generate corrected answer
        try:
            corrected_answer = self.model_manager.generate_answer(
                prompt,
                max_length=150
            )
            
            logger.info("Generated corrected answer")
            return corrected_answer
            
        except Exception as e:
            logger.error(f"Error generating corrected answer: {e}")
            return original_answer
    
    def _format_evidence(self, evidence: List[Dict[str, any]], max_passages: int = 3) -> str:
        """
        Format evidence passages into a readable context string.
        
        Args:
            evidence: List of evidence dictionaries
            max_passages: Maximum number of passages to include
            
        Returns:
            Formatted evidence string
        """
        # Take top passages by score
        sorted_evidence = sorted(evidence, key=lambda x: x.get("score", 0), reverse=True)
        top_evidence = sorted_evidence[:max_passages]
        
        formatted = []
        for i, ev in enumerate(top_evidence, 1):
            snippet = ev["snippet"][:300]  # Truncate long passages
            formatted.append(f"[{i}] {snippet}")
        
        return "\n".join(formatted)
    
    def _create_correction_prompt(
        self,
        question: str,
        evidence_context: str,
        low_confidence_claims: List[str] = None
    ) -> str:
        """
        Create a prompt for evidence-grounded answer generation.
        
        Args:
            question: Original question
            evidence_context: Formatted evidence passages
            low_confidence_claims: Claims needing correction
            
        Returns:
            Prompt string
        """
        # Base prompt with evidence grounding
        prompt = (
            f"Based on the following evidence, answer the question accurately. "
            f"Only use information from the evidence provided.\n\n"
            f"Evidence:\n{evidence_context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        
        return prompt
    
    def generate_grounded_answer(
        self,
        question: str,
        evidence: List[Dict[str, any]]
    ) -> str:
        """
        Generate an answer directly from evidence (alternative approach).
        
        Args:
            question: Question to answer
            evidence: Retrieved evidence
            
        Returns:
            Evidence-grounded answer
        """
        evidence_context = self._format_evidence(evidence)
        
        prompt = (
            f"Using only the information below, provide a concise answer.\n\n"
            f"Information:\n{evidence_context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the information:"
        )
        
        answer = self.model_manager.generate_answer(prompt, max_length=150)
        return answer
