"""
FastAPI Application - Hallucination Detection System
Main API endpoints for the research system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os

from models import get_model_manager
from retrieval import RetrievalSystem, create_sample_corpus
from hallucination import HallucinationDetector
from correction import AnswerCorrector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lightweight Hallucination Detection System",
    description="Research system for detecting and reducing hallucinations in local LLMs",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system components
model_manager = None
retrieval_system = None
hallucination_detector = None
answer_corrector = None


# Request/Response Models
class QuestionRequest(BaseModel):
    question: str


class EvidenceResponse(BaseModel):
    source: str
    snippet: str
    score: Optional[float] = None


class AnswerResponse(BaseModel):
    answer: str
    is_hallucinated: bool
    hallucination_score: float
    corrected_answer: Optional[str] = None
    evidence: List[EvidenceResponse]
    claims: Optional[List[str]] = None
    claim_scores: Optional[List[float]] = None


class EvaluationRequest(BaseModel):
    dataset_file: str


class EvaluationResponse(BaseModel):
    hallucination_rate: float
    average_score: float
    accuracy_before: float
    accuracy_after: float
    total_questions: int


@app.on_event("startup")
async def startup_event():
    """
    Initialize all system components on startup.
    """
    global model_manager, retrieval_system, hallucination_detector, answer_corrector
    
    logger.info("=" * 60)
    logger.info("Starting Hallucination Detection System")
    logger.info("=" * 60)
    
    try:
        # Initialize models
        logger.info("Step 1/4: Loading models...")
        model_manager = get_model_manager()
        
        # Initialize retrieval system
        logger.info("Step 2/4: Building retrieval index...")
        retrieval_system = RetrievalSystem(model_manager.embedding_model)
        
        # Load or create corpus
        corpus_path = "data/corpus.txt"
        if os.path.exists(corpus_path):
            corpus_texts = create_sample_corpus(corpus_path)
        else:
            logger.warning(f"Corpus file not found at {corpus_path}, using sample data")
            corpus_texts = get_sample_corpus_data()
            # Save sample corpus
            os.makedirs("data", exist_ok=True)
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(corpus_texts))
        
        if corpus_texts:
            retrieval_system.build_index(corpus_texts)
        else:
            logger.warning("No corpus data available. Retrieval may not work properly.")
        
        # Initialize hallucination detector
        logger.info("Step 3/4: Initializing hallucination detector...")
        hallucination_detector = HallucinationDetector(
            model_manager.embedding_model,
            threshold=0.45
        )
        
        # Initialize answer corrector
        logger.info("Step 4/4: Initializing answer corrector...")
        answer_corrector = AnswerCorrector(model_manager)
        
        logger.info("=" * 60)
        logger.info("✓ System ready!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise


@app.get("/")
async def root():
    """
    Root endpoint - system status.
    """
    return {
        "status": "online",
        "system": "Lightweight Hallucination Detection",
        "version": "1.0.0",
        "models": {
            "llm": "google/flan-t5-small",
            "embedding": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint: Answer a question with hallucination detection.
    
    Process:
    1. Generate answer using LLM
    2. Retrieve relevant evidence from corpus
    3. Detect hallucinations via claim verification
    4. Correct answer if hallucination detected
    """
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing question: {question[:100]}...")
        
        # Step 1: Generate initial answer
        logger.info("Generating answer...")
        answer = model_manager.generate_answer(question)
        
        # Step 2: Retrieve evidence
        logger.info("Retrieving evidence...")
        evidence = retrieval_system.retrieve(question, top_k=5)
        
        # Step 3: Detect hallucination
        logger.info("Detecting hallucinations...")
        detection_result = hallucination_detector.detect_hallucination(answer, evidence)
        
        # Step 4: Correct if needed
        corrected_answer = None
        if detection_result["is_hallucinated"]:
            logger.info("Hallucination detected, generating correction...")
            corrected_answer = answer_corrector.correct_answer(
                question,
                answer,
                evidence,
                detection_result.get("low_confidence_claims", [])
            )
        
        # Format response
        response = AnswerResponse(
            answer=answer,
            is_hallucinated=detection_result["is_hallucinated"],
            hallucination_score=detection_result["hallucination_score"],
            corrected_answer=corrected_answer,
            evidence=[
                EvidenceResponse(
                    source=e["source"],
                    snippet=e["snippet"][:500],  # Truncate for response
                    score=e.get("score")
                )
                for e in evidence[:3]  # Return top 3 evidence
            ],
            claims=detection_result["claims"],
            claim_scores=detection_result["claim_scores"]
        )
        
        logger.info(f"Request completed: hallucinated={response.is_hallucinated}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "models_loaded": model_manager is not None,
        "retrieval_ready": retrieval_system is not None and retrieval_system.index is not None
    }


def get_sample_corpus_data() -> List[str]:
    """
    Generate sample corpus data for demonstration.
    Replace this with real data from Wikipedia, papers, etc.
    """
    return [
        "The Earth is the third planet from the Sun and the only astronomical object known to harbor life. "
        "About 71% of Earth's surface is covered with water, mostly by oceans. Earth's atmosphere consists "
        "mainly of nitrogen and oxygen.",
        
        "Python is a high-level, interpreted programming language created by Guido van Rossum and first "
        "released in 1991. Python's design philosophy emphasizes code readability with the use of significant indentation.",
        
        "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on "
        "the use of data and algorithms to imitate the way that humans learn, gradually improving accuracy over time.",
        
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named "
        "after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.",
        
        "Photosynthesis is the process by which green plants and certain other organisms transform light energy "
        "into chemical energy. During photosynthesis in green plants, light energy is captured and used to convert "
        "water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.",
        
        "The theory of relativity, developed by Albert Einstein, encompasses two interrelated theories: special "
        "relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity. "
        "General relativity explains the law of gravitation and its relation to other forces of nature.",
        
        "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other "
        "to form a double helix. DNA carries genetic instructions for the development, functioning, growth and "
        "reproduction of all known organisms and many viruses.",
        
        "Climate change refers to long-term shifts in temperatures and weather patterns. Since the 1800s, human "
        "activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, "
        "oil and gas, which produces heat-trapping gases."
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
