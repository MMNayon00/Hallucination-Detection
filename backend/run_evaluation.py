"""
Run Evaluation Script
Executes the evaluation pipeline and generates results
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from models import get_model_manager
from retrieval import RetrievalSystem, create_sample_corpus
from hallucination import HallucinationDetector
from correction import AnswerCorrector
from evaluation import SystemEvaluator, create_sample_dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main evaluation execution function.
    """
    logger.info("="*70)
    logger.info("HALLUCINATION DETECTION SYSTEM - EVALUATION PIPELINE")
    logger.info("="*70)
    
    # Step 1: Initialize models
    logger.info("\n[1/6] Loading models...")
    model_manager = get_model_manager()
    
    # Step 2: Build retrieval index
    logger.info("\n[2/6] Building retrieval index...")
    retrieval_system = RetrievalSystem(model_manager.embedding_model)
    
    corpus_path = "data/corpus.txt"
    if not os.path.exists(corpus_path):
        logger.warning(f"Corpus not found at {corpus_path}")
        return
    
    corpus_texts = create_sample_corpus(corpus_path)
    retrieval_system.build_index(corpus_texts)
    
    # Step 3: Initialize detection and correction
    logger.info("\n[3/6] Initializing hallucination detector and corrector...")
    hallucination_detector = HallucinationDetector(
        model_manager.embedding_model,
        threshold=0.45
    )
    answer_corrector = AnswerCorrector(model_manager)
    
    # Step 4: Load or create evaluation dataset
    logger.info("\n[4/6] Loading evaluation dataset...")
    dataset_path = "data/qa_dataset.jsonl"
    
    if not os.path.exists(dataset_path):
        logger.info("Creating sample dataset...")
        create_sample_dataset(dataset_path)
    
    # Step 5: Create evaluator and run evaluation
    logger.info("\n[5/6] Running evaluation...")
    evaluator = SystemEvaluator(
        model_manager,
        retrieval_system,
        hallucination_detector,
        answer_corrector
    )
    
    dataset = evaluator.load_dataset(dataset_path)
    metrics = evaluator.evaluate_dataset(dataset)
    
    # Step 6: Save results and generate plots
    logger.info("\n[6/6] Generating results and visualizations...")
    
    # Create results directory
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    evaluator.save_results(results_dir)
    evaluator.plot_results(results_dir)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE - RESULTS SUMMARY")
    logger.info("="*70)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key:.<40} {value:.4f}")
        else:
            logger.info(f"{key:.<40} {value}")
    
    logger.info("\n" + "="*70)
    logger.info(f"Results saved to: {results_dir}/")
    logger.info("  - evaluation_results.csv (detailed results)")
    logger.info("  - metrics.json (aggregated metrics)")
    logger.info("  - results_table.tex (LaTeX table)")
    logger.info("  - score_distribution.png (visualization)")
    logger.info("  - accuracy_comparison.png (visualization)")
    logger.info("  - roc_curve.png (if applicable)")
    logger.info("="*70)


if __name__ == "__main__":
    main()
