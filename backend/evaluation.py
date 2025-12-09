"""
Evaluation Pipeline for Academic Publication
Evaluates system performance on benchmark datasets with scientific metrics
"""

import json
import jsonlines
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemEvaluator:
    """
    Comprehensive evaluation pipeline for hallucination detection system.
    Produces publication-ready metrics, tables, and visualizations.
    """
    
    def __init__(self, model_manager, retrieval_system, hallucination_detector, answer_corrector):
        """
        Initialize evaluator with system components.
        """
        self.model_manager = model_manager
        self.retrieval_system = retrieval_system
        self.hallucination_detector = hallucination_detector
        self.answer_corrector = answer_corrector
        
        # Results storage
        self.results = []
        
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load evaluation dataset from JSONL file.
        
        Expected format:
        {"question": "...", "answer": "...", "is_hallucinated": true/false}
        or
        {"question": "...", "ground_truth": "..."}
        
        Args:
            dataset_path: Path to JSONL dataset file
            
        Returns:
            List of dataset items
        """
        dataset = []
        
        try:
            with jsonlines.open(dataset_path) as reader:
                for item in reader:
                    dataset.append(item)
            logger.info(f"Loaded {len(dataset)} items from {dataset_path}")
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            raise
        
        return dataset
    
    def evaluate_single_question(
        self,
        question: str,
        ground_truth: str = None,
        is_hallucinated_label: bool = None
    ) -> Dict:
        """
        Evaluate system on a single question.
        
        Args:
            question: Question text
            ground_truth: Optional ground truth answer
            is_hallucinated_label: Optional ground truth hallucination label
            
        Returns:
            Dictionary with evaluation results
        """
        # Generate answer
        answer = self.model_manager.generate_answer(question)
        
        # Retrieve evidence
        evidence = self.retrieval_system.retrieve(question, top_k=5)
        
        # Detect hallucination
        detection_result = self.hallucination_detector.detect_hallucination(answer, evidence)
        
        # Correct if needed
        corrected_answer = None
        if detection_result["is_hallucinated"]:
            corrected_answer = self.answer_corrector.correct_answer(
                question, answer, evidence, detection_result.get("low_confidence_claims", [])
            )
        
        # Compute semantic similarity with ground truth if available
        answer_similarity = None
        corrected_similarity = None
        
        if ground_truth:
            # Encode answers and ground truth
            embeddings = self.model_manager.encode_text([answer, ground_truth])
            answer_similarity = float(self._cosine_similarity(embeddings[0], embeddings[1]))
            
            if corrected_answer:
                embeddings = self.model_manager.encode_text([corrected_answer, ground_truth])
                corrected_similarity = float(self._cosine_similarity(embeddings[0], embeddings[1]))
        
        result = {
            "question": question,
            "answer": answer,
            "corrected_answer": corrected_answer,
            "ground_truth": ground_truth,
            "is_hallucinated_pred": detection_result["is_hallucinated"],
            "is_hallucinated_label": is_hallucinated_label,
            "hallucination_score": detection_result["hallucination_score"],
            "answer_similarity": answer_similarity,
            "corrected_similarity": corrected_similarity,
            "num_claims": len(detection_result["claims"]),
            "mean_claim_score": np.mean(detection_result["claim_scores"]) if detection_result["claim_scores"] else 0.0
        }
        
        return result
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """
        Evaluate system on full dataset.
        
        Args:
            dataset: List of dataset items
            
        Returns:
            Aggregated evaluation metrics
        """
        logger.info(f"Evaluating on {len(dataset)} questions...")
        
        self.results = []
        
        for item in tqdm(dataset, desc="Evaluating"):
            question = item.get("question", "")
            ground_truth = item.get("ground_truth") or item.get("answer")
            is_hallucinated_label = item.get("is_hallucinated")
            
            result = self.evaluate_single_question(
                question,
                ground_truth,
                is_hallucinated_label
            )
            
            self.results.append(result)
        
        # Compute aggregated metrics
        metrics = self._compute_metrics()
        
        logger.info("Evaluation complete!")
        return metrics
    
    def _compute_metrics(self) -> Dict:
        """
        Compute aggregated metrics from evaluation results.
        
        Returns:
            Dictionary of metrics
        """
        df = pd.DataFrame(self.results)
        
        metrics = {
            "total_questions": len(self.results),
            "average_hallucination_score": float(df["hallucination_score"].mean()),
            "hallucination_rate": float(df["is_hallucinated_pred"].mean()),
        }
        
        # Hallucination detection metrics (if labels available)
        if "is_hallucinated_label" in df.columns and df["is_hallucinated_label"].notna().any():
            valid_labels = df["is_hallucinated_label"].notna()
            y_true = df.loc[valid_labels, "is_hallucinated_label"].astype(int)
            y_pred = df.loc[valid_labels, "is_hallucinated_pred"].astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            
            metrics.update({
                "detection_precision": float(precision),
                "detection_recall": float(recall),
                "detection_f1": float(f1),
                "detection_accuracy": float(accuracy_score(y_true, y_pred))
            })
        
        # Answer quality metrics (if ground truth available)
        if "answer_similarity" in df.columns and df["answer_similarity"].notna().any():
            valid_sims = df["answer_similarity"].notna()
            
            metrics.update({
                "accuracy_before": float(df.loc[valid_sims, "answer_similarity"].mean()),
                "answer_similarity_std": float(df.loc[valid_sims, "answer_similarity"].std())
            })
            
            # Corrected answer metrics
            if "corrected_similarity" in df.columns:
                valid_corrected = df["corrected_similarity"].notna()
                if valid_corrected.any():
                    metrics["accuracy_after"] = float(df.loc[valid_corrected, "corrected_similarity"].mean())
                    
                    # Improvement
                    improved_mask = valid_corrected & (df["corrected_similarity"] > df["answer_similarity"])
                    metrics["improvement_rate"] = float(improved_mask.sum() / valid_corrected.sum())
        
        return metrics
    
    def save_results(self, output_dir: str = "results"):
        """
        Save evaluation results to CSV and JSON files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as CSV
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to {csv_path}")
        
        # Save metrics as JSON
        metrics = self._compute_metrics()
        json_path = os.path.join(output_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {json_path}")
        
        # Save for LaTeX table
        self._save_latex_table(metrics, output_dir)
        
    def _save_latex_table(self, metrics: Dict, output_dir: str):
        """
        Generate LaTeX table from metrics.
        """
        latex_path = os.path.join(output_dir, "results_table.tex")
        
        with open(latex_path, "w") as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\hline\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\hline\n")
            
            for key, value in metrics.items():
                formatted_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    f.write(f"{formatted_key} & {value:.4f} \\\\\n")
                else:
                    f.write(f"{formatted_key} & {value} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Hallucination Detection System Performance}\n")
            f.write("\\label{tab:results}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"Saved LaTeX table to {latex_path}")
    
    def plot_results(self, output_dir: str = "results"):
        """
        Generate publication-quality plots.
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        # Set style for publication
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        # Plot 1: Hallucination score distribution
        self._plot_score_distribution(df, output_dir)
        
        # Plot 2: Accuracy comparison (before vs after)
        if "answer_similarity" in df.columns and df["answer_similarity"].notna().any():
            self._plot_accuracy_comparison(df, output_dir)
        
        # Plot 3: ROC curve (if labels available)
        if "is_hallucinated_label" in df.columns and df["is_hallucinated_label"].notna().any():
            self._plot_roc_curve(df, output_dir)
        
        logger.info(f"Saved plots to {output_dir}")
    
    def _plot_score_distribution(self, df: pd.DataFrame, output_dir: str):
        """Plot hallucination score distribution."""
        plt.figure(figsize=(8, 6))
        plt.hist(df["hallucination_score"], bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(self.hallucination_detector.threshold, color="red", linestyle="--", 
                   label=f"Threshold = {self.hallucination_detector.threshold}")
        plt.xlabel("Hallucination Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Distribution of Hallucination Scores", fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "score_distribution.png"))
        plt.close()
    
    def _plot_accuracy_comparison(self, df: pd.DataFrame, output_dir: str):
        """Plot accuracy before and after correction."""
        valid_data = df[df["corrected_similarity"].notna()]
        
        if len(valid_data) == 0:
            return
        
        plt.figure(figsize=(8, 6))
        
        metrics = ["Before Correction", "After Correction"]
        values = [
            df["answer_similarity"].mean(),
            valid_data["corrected_similarity"].mean()
        ]
        
        bars = plt.bar(metrics, values, color=["#ff7f0e", "#2ca02c"], alpha=0.8, edgecolor="black")
        plt.ylabel("Average Similarity to Ground Truth", fontsize=12)
        plt.title("Answer Quality: Before vs After Correction", fontsize=14)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
        plt.close()
    
    def _plot_roc_curve(self, df: pd.DataFrame, output_dir: str):
        """Plot ROC curve for hallucination detection."""
        valid_labels = df["is_hallucinated_label"].notna()
        y_true = df.loc[valid_labels, "is_hallucinated_label"].astype(int)
        y_scores = df.loc[valid_labels, "hallucination_score"]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve - Hallucination Detection", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)


def create_sample_dataset(output_path: str):
    """
    Create a sample evaluation dataset for testing.
    
    Args:
        output_path: Path to save JSONL dataset
    """
    sample_data = [
        {
            "question": "What is the capital of France?",
            "ground_truth": "The capital of France is Paris.",
            "is_hallucinated": False
        },
        {
            "question": "Who invented Python programming language?",
            "ground_truth": "Python was created by Guido van Rossum and first released in 1991.",
            "is_hallucinated": False
        },
        {
            "question": "What percentage of Earth is covered by water?",
            "ground_truth": "About 71% of Earth's surface is covered with water.",
            "is_hallucinated": False
        },
        {
            "question": "When was the Eiffel Tower built?",
            "ground_truth": "The Eiffel Tower was built from 1887 to 1889.",
            "is_hallucinated": False
        },
        {
            "question": "What is photosynthesis?",
            "ground_truth": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "is_hallucinated": False
        }
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(sample_data)
    
    logger.info(f"Created sample dataset at {output_path}")


if __name__ == "__main__":
    # Create sample dataset if it doesn't exist
    dataset_path = "data/qa_dataset.jsonl"
    if not os.path.exists(dataset_path):
        create_sample_dataset(dataset_path)
