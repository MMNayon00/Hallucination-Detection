# Lightweight Hallucination Detection System

> **Research Project:** Lightweight Hallucination Detection and Reduction in Local LLM Systems using Retrieval-Based Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, publishable academic research system for detecting and reducing hallucinations in small language models. Optimized for CPU-only inference on resource-constrained devices (MacBook Air M2, 8GB RAM).

---

## 🎯 Project Overview

This system implements a novel approach to hallucination detection using:

- **Small Models:** google/flan-t5-small (77M params) + sentence-transformers/all-MiniLM-L6-v2 (22M params)
- **Retrieval-Based Verification:** FAISS vector search for evidence retrieval
- **Claim-Level Scoring:** Semantic similarity between claims and evidence
- **Evidence-Grounded Correction:** Automatic answer correction using retrieved facts
- **Publication-Ready Evaluation:** Comprehensive metrics, visualizations, and LaTeX tables

### Key Features

✅ **CPU-Friendly:** Runs on MacBook Air M2 with 8GB RAM (no GPU required)  
✅ **Fast Inference:** Average response time < 5 seconds  
✅ **Interpretable:** Claim-level hallucination scoring with evidence attribution  
✅ **Reproducible:** Fixed seeds, documented hyperparameters, evaluation pipeline  
✅ **Complete:** Backend API + Frontend demo + Evaluation tools  

---

## 📁 Project Structure

```
Hallucination Detection/
├── backend/
│   ├── app.py                 # FastAPI server
│   ├── models.py              # Model loading (T5 + embeddings)
│   ├── retrieval.py           # FAISS-based vector retrieval
│   ├── hallucination.py       # Detection & scoring algorithm
│   ├── correction.py          # Evidence-grounded answer correction
│   ├── evaluation.py          # Academic evaluation pipeline
│   ├── run_evaluation.py      # Evaluation script
│   ├── requirements.txt       # Python dependencies
│   └── data/
│       ├── corpus.txt         # Evidence corpus (Wikipedia excerpts)
│       └── qa_dataset.jsonl   # Evaluation QA dataset
│
├── frontend/
│   ├── index.html             # Research demo interface
│   ├── styles.css             # Modern UI styling
│   └── script.js              # API integration
│
├── results/                   # Generated evaluation outputs
│   ├── evaluation_results.csv
│   ├── metrics.json
│   ├── results_table.tex
│   └── *.png (visualizations)
│
├── paper/
│   └── research_paper_template.md  # Publication template
│
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- 8GB+ RAM
- ~2GB disk space for models and data

### Installation

1. **Clone or navigate to the project:**
   ```bash
   cd "Hallucination Detection/backend"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Note: First run will download models (~400MB total). This may take 5-10 minutes.*

### Running the System

#### Option 1: API Server + Frontend Demo

1. **Start the backend server:**
   ```bash
   python app.py
   ```
   
   Server will start at `http://localhost:8000`

2. **Open the frontend:**
   ```bash
   # In a new terminal, from the project root:
   cd frontend
   open index.html  # macOS
   # or just open index.html in your browser
   ```

3. **Try it out:**
   - Enter a question in the UI
   - Click "Analyze Question"
   - View hallucination detection results, scores, and evidence

#### Option 2: Run Evaluation Pipeline

Generate publication-ready results:

```bash
cd backend
python run_evaluation.py
```

This will:
- Load models and build retrieval index
- Evaluate on the QA dataset
- Generate metrics (precision, recall, F1, accuracy)
- Create visualizations (ROC curve, score distribution, accuracy comparison)
- Save results to `results/` directory

---

## 📊 System Workflow

```
User Question
     ↓
[LLM Generation] → Generated Answer
     ↓
[Retrieval] → Top-K Evidence Passages
     ↓
[Claim Extraction] → Individual Claims
     ↓
[Semantic Similarity] → Claim Scores
     ↓
[Aggregation] → Hallucination Score
     ↓
[Classification] → Hallucinated? (Yes/No)
     ↓ (if Yes)
[Evidence-Grounded Correction] → Corrected Answer
```

---

## 🔬 Scientific Methodology

### Hallucination Detection Algorithm

1. **Claim Extraction:** Segment answer into individual claims (sentences)
2. **Evidence Encoding:** Encode retrieved passages using sentence-transformers
3. **Similarity Scoring:** Compute cosine similarity between each claim and all evidence
4. **Claim Score:** `s(claim) = max(similarity to evidence passages)`
5. **Hallucination Score:** `H = 1 - mean(claim_scores)`
6. **Classification:** `hallucinated = (H > 0.45)`

### Models Used

| Component | Model | Parameters | Size |
|-----------|-------|-----------|------|
| LLM | google/flan-t5-small | 77M | ~300MB |
| Embeddings | all-MiniLM-L6-v2 | 22M | ~90MB |
| Vector Store | FAISS (CPU) | - | In-memory |

### Evaluation Metrics

- **Detection:** Precision, Recall, F1, Accuracy, ROC-AUC
- **Answer Quality:** Semantic similarity to ground truth
- **Improvement:** Before/after correction accuracy
- **Efficiency:** Inference time, memory usage

---

## 📖 API Documentation

### `POST /ask`

Analyze a question with hallucination detection.

**Request:**
```json
{
  "question": "What is the capital of France?"
}
```

**Response:**
```json
{
  "answer": "The capital of France is Paris.",
  "is_hallucinated": false,
  "hallucination_score": 0.12,
  "corrected_answer": null,
  "evidence": [
    {
      "source": "corpus_passage_42",
      "snippet": "Paris is the capital and largest city of France...",
      "score": 0.94
    }
  ],
  "claims": ["The capital of France is Paris."],
  "claim_scores": [0.88]
}
```

### `GET /health`

Check system health.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "retrieval_ready": true
}
```

---

## 🎨 Frontend Features

The research demo interface provides:

- **Question Input:** Text area with sample questions
- **Hallucination Alert:** Color-coded warning banner (green/yellow/red)
- **Score Display:** Visual gauge showing hallucination probability
- **Answer Comparison:** Side-by-side original vs. corrected answers
- **Claim Analysis:** Breakdown of individual claim verification scores
- **Evidence Display:** Retrieved passages with relevance scores
- **Technical Details:** Collapsible section with raw metrics

---

## 📈 Evaluation Pipeline

### Running Experiments

```bash
cd backend
python run_evaluation.py
```

### Generated Outputs

1. **`results/evaluation_results.csv`**
   - Per-question detailed results
   - Columns: question, answer, hallucination_score, similarity, etc.

2. **`results/metrics.json`**
   - Aggregated metrics for paper
   - Detection accuracy, improvement rates

3. **`results/results_table.tex`**
   - LaTeX table ready for paper inclusion

4. **Visualizations:**
   - `score_distribution.png` - Histogram of hallucination scores
   - `accuracy_comparison.png` - Before/after correction bar chart
   - `roc_curve.png` - ROC curve for detection (if labels available)

---

## 🔧 Configuration

### Adjusting Hallucination Threshold

In `backend/hallucination.py`, modify the threshold:

```python
hallucination_detector = HallucinationDetector(
    embedding_model,
    threshold=0.45  # Increase for stricter detection
)
```

### Changing Retrieval Parameters

In `backend/app.py`, adjust top-k retrieval:

```python
evidence = retrieval_system.retrieve(question, top_k=5)  # Retrieve more passages
```

### Expanding the Corpus

Add more text to `backend/data/corpus.txt`:

```
Your factual paragraphs here...

Separate with double newlines.

More information...
```

---

## 📝 Research Paper

A complete research paper template is provided in `paper/research_paper_template.md` with sections for:

- Abstract
- Introduction & Research Questions
- Related Work
- Methodology (algorithm description)
- Experimental Setup
- Results (tables and figures)
- Discussion & Limitations
- Future Work
- Appendices (implementation details)

**To use:** Fill in the `[TO BE FILLED]` sections with your experimental results.

---

## 🎓 Academic Use

This project is designed for:

✅ **NLP Research:** Study hallucination detection methods  
✅ **Course Projects:** Demonstrate RAG and verification techniques  
✅ **Publications:** Template ready for conference/journal submission  
✅ **Education:** Learn about LLMs, retrieval, and evaluation  

### Citation

If you use this work, please cite:

```bibtex
@misc{hallucination_detection_2024,
  title={Lightweight Hallucination Detection and Reduction in Local LLM Systems},
  author={[Your Name]},
  year={2024},
  note={Research project for [Course/Institution]}
}
```

---

## 🛠 Troubleshooting

### Models Not Loading
- **Issue:** HuggingFace models fail to download
- **Solution:** Check internet connection, ensure ~2GB free disk space

### Memory Errors
- **Issue:** System runs out of RAM
- **Solution:** Close other applications, reduce batch size in `models.py`

### Slow Inference
- **Issue:** Responses take >10 seconds
- **Solution:** Normal for first run (model loading). Subsequent queries are faster.

### CORS Errors (Frontend)
- **Issue:** Browser blocks API requests
- **Solution:** Ensure backend is running, or use `python -m http.server` to serve frontend

---

## 🚧 Limitations

1. **Corpus Dependency:** Detection quality depends on corpus coverage
2. **Small Model Constraints:** May miss complex, nuanced hallucinations
3. **Language Support:** Optimized for English text
4. **Latency:** Retrieval adds ~2-3 seconds per query
5. **Binary Classification:** Threshold-based detection (not probabilistic)

---

## 🔮 Future Enhancements

- [ ] Multi-language support (multilingual embeddings)
- [ ] Fine-tuned small models on fact-checking datasets
- [ ] Dynamic corpus expansion with web search
- [ ] Uncertainty quantification (Bayesian scoring)
- [ ] Streaming responses with real-time detection
- [ ] GPU acceleration option for faster inference

---

## 📄 License

This project is released under the MIT License. Free to use for academic and research purposes.

---

## 🙏 Acknowledgments

**Models:**
- T5: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- Sentence-Transformers: Reimers & Gurevych, "Sentence-BERT"

**Frameworks:**
- FastAPI, PyTorch, HuggingFace Transformers, FAISS

---

**Built with ❤️ for reproducible AI research**
