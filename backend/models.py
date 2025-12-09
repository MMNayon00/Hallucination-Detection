"""
Model Loading Module for Lightweight Hallucination Detection
Optimized for MacBook Air M2 with 8GB RAM
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading and inference for small, CPU-friendly models.
    
    Models used:
    - LLM: google/flan-t5-small (77M parameters) - CPU-friendly text generation
    - Embedding: sentence-transformers/all-MiniLM-L6-v2 (22M parameters) - semantic encoding
    """
    
    def __init__(self):
        self.device = "cpu"  # Force CPU for M2 compatibility
        self.llm_model = None
        self.llm_tokenizer = None
        self.embedding_model = None
        
    def load_llm(self, model_name: str = "google/flan-t5-small"):
        """
        Load the small generative LLM for answer generation.
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading LLM: {model_name}")
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
            self.llm_model.to(self.device)
            self.llm_model.eval()  # Set to evaluation mode
            logger.info(f"✓ LLM loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
    
    def load_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load the sentence embedding model for retrieval and verification.
        
        Args:
            model_name: Sentence-transformers model identifier
        """
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"✓ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_answer(self, question: str, max_length: int = 128) -> str:
        """
        Generate an answer to a question using the LLM.
        
        Args:
            question: Input question string
            max_length: Maximum tokens in generated answer
            
        Returns:
            Generated answer string
        """
        if self.llm_model is None or self.llm_tokenizer is None:
            raise RuntimeError("LLM not loaded. Call load_llm() first.")
        
        # Format the prompt for T5
        prompt = f"Answer the following question: {question}"
        
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,  # Beam search for better quality
                early_stopping=True,
                temperature=0.7,
                do_sample=False  # Deterministic for reproducibility
            )
        
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Encode text into semantic embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Tensor of embeddings (shape: [len(texts), embedding_dim])
        """
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Call load_embedding_model() first.")
        
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=8  # Small batch size for memory efficiency
        )
        return embeddings
    
    def initialize_all_models(self):
        """
        Convenience method to load all models at once.
        """
        logger.info("Initializing all models...")
        self.load_llm()
        self.load_embedding_model()
        logger.info("✓ All models initialized successfully")


# Singleton instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """
    Get the singleton ModelManager instance.
    
    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
        _model_manager.initialize_all_models()
    return _model_manager
