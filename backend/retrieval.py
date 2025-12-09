"""
Retrieval Module - FAISS-based Vector Search
Retrieves relevant evidence from local corpus for hallucination verification
"""

import numpy as np
import faiss
from typing import List, Dict, Tuple
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalSystem:
    """
    Vector-based retrieval system using FAISS for efficient similarity search.
    Stores and retrieves evidence passages from a local corpus.
    """
    
    def __init__(self, embedding_model):
        """
        Initialize the retrieval system.
        
        Args:
            embedding_model: The embedding model from ModelManager
        """
        self.embedding_model = embedding_model
        self.index = None
        self.passages = []
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
    def build_index(self, corpus_texts: List[str], chunk_size: int = 200):
        """
        Build FAISS index from corpus texts.
        
        Args:
            corpus_texts: List of text passages to index
            chunk_size: Characters per chunk when splitting long texts
        """
        logger.info(f"Building index from {len(corpus_texts)} documents...")
        
        # Chunk long texts into passages
        self.passages = []
        for text in corpus_texts:
            if len(text) > chunk_size:
                # Split into chunks with overlap
                chunks = self._chunk_text(text, chunk_size, overlap=50)
                self.passages.extend(chunks)
            else:
                self.passages.append(text)
        
        logger.info(f"Created {len(self.passages)} passages from corpus")
        
        # Generate embeddings for all passages
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            self.passages,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=8
        )
        
        # Convert to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Build FAISS index (using L2 distance, which works well with normalized embeddings)
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity via L2 distance
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        logger.info(f"✓ Index built with {self.index.ntotal} vectors")
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += (chunk_size - overlap)
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Retrieve top-k most relevant passages for a query.
        
        Args:
            query: Query text
            top_k: Number of passages to retrieve
            
        Returns:
            List of dictionaries with 'snippet' and 'score' keys
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_tensor=False,
            show_progress_bar=False
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Convert L2 distances to similarity scores
        # After normalization, L2 distance d relates to cosine similarity as: sim = 1 - d^2/2
        # For simplicity, we'll use 1/(1+distance) as similarity score
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.passages):  # Valid index
                similarity_score = 1.0 / (1.0 + dist)
                results.append({
                    "snippet": self.passages[idx],
                    "score": float(similarity_score),
                    "source": f"corpus_passage_{idx}"
                })
        
        return results
    
    def save_index(self, save_path: str):
        """
        Save FAISS index and passages to disk.
        
        Args:
            save_path: Directory path to save index
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_path, "faiss_index.bin"))
        
        # Save passages
        with open(os.path.join(save_path, "passages.pkl"), "wb") as f:
            pickle.dump(self.passages, f)
        
        logger.info(f"✓ Index saved to {save_path}")
    
    def load_index(self, load_path: str):
        """
        Load FAISS index and passages from disk.
        
        Args:
            load_path: Directory path to load index from
        """
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))
        
        # Load passages
        with open(os.path.join(load_path, "passages.pkl"), "rb") as f:
            self.passages = pickle.load(f)
        
        logger.info(f"✓ Index loaded from {load_path}")


def create_sample_corpus(file_path: str) -> List[str]:
    """
    Create or load a sample corpus from a text file.
    
    Args:
        file_path: Path to corpus file
        
    Returns:
        List of text documents
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by double newlines (paragraphs)
        documents = [doc.strip() for doc in content.split("\n\n") if doc.strip()]
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    else:
        logger.warning(f"Corpus file not found: {file_path}")
        return []
