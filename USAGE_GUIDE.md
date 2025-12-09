# Development and Usage Guide

## Quick Reference Commands

### Setup (First Time Only)

```bash
# From project root
chmod +x setup.sh
./setup.sh
```

### Running the System

#### Start Backend Server
```bash
cd backend
source venv/bin/activate  # Activate virtual environment
python app.py
```

Server will be available at: `http://localhost:8000`

#### Access Frontend
Open `frontend/index.html` in your web browser, or:

```bash
cd frontend
python -m http.server 8080
# Then visit http://localhost:8080
```

#### Run Evaluation
```bash
cd backend
source venv/bin/activate
python run_evaluation.py
```

---

## API Testing with cURL

### Test Health Endpoint
```bash
curl http://localhost:8000/health
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

### Pretty Print Response
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who created Python?"}'| python -m json.tool
```

---

## Python API Usage

### Direct Model Usage

```python
from models import get_model_manager

# Load models
manager = get_model_manager()

# Generate answer
answer = manager.generate_answer("What is photosynthesis?")
print(answer)

# Encode text
embeddings = manager.encode_text([
    "This is a sentence.",
    "This is another sentence."
])
print(embeddings.shape)
```

### Retrieval System

```python
from retrieval import RetrievalSystem, create_sample_corpus
from models import get_model_manager

# Setup
manager = get_model_manager()
retrieval = RetrievalSystem(manager.embedding_model)

# Load corpus
corpus = create_sample_corpus("data/corpus.txt")
retrieval.build_index(corpus)

# Search
results = retrieval.retrieve("Python programming language", top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f} - {r['snippet'][:100]}")
```

### Hallucination Detection

```python
from hallucination import HallucinationDetector
from models import get_model_manager

manager = get_model_manager()
detector = HallucinationDetector(manager.embedding_model, threshold=0.45)

answer = "Paris is the capital of France and was founded in 1800."
evidence = [
    {"snippet": "Paris is the capital and largest city of France."}
]

result = detector.detect_hallucination(answer, evidence)
print(f"Hallucinated: {result['is_hallucinated']}")
print(f"Score: {result['hallucination_score']:.3f}")
print(f"Claims: {result['claims']}")
```

---

## Extending the System

### Adding New Corpus Data

1. Edit `backend/data/corpus.txt`
2. Add new paragraphs (separated by double newlines)
3. Restart the server (it rebuilds the index on startup)

Example:
```
Quantum computing uses quantum bits (qubits) that can exist in superposition.
Unlike classical bits, qubits can be 0, 1, or both simultaneously.

The Great Barrier Reef is the world's largest coral reef system.
It is located off the coast of Queensland, Australia.
```

### Creating Custom Evaluation Datasets

Edit `backend/data/qa_dataset.jsonl`:

```jsonl
{"question": "Your question?", "ground_truth": "Expected answer", "is_hallucinated": false}
{"question": "Another question?", "ground_truth": "Another answer", "is_hallucinated": false}
```

### Adjusting Detection Sensitivity

In `backend/app.py`, modify the threshold:

```python
hallucination_detector = HallucinationDetector(
    model_manager.embedding_model,
    threshold=0.45  # Lower = stricter, Higher = more lenient
)
```

Recommended values:
- `0.30` - Very strict (high false positives)
- `0.45` - Balanced (default)
- `0.60` - Lenient (may miss hallucinations)

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Activate virtual environment
```bash
cd backend
source venv/bin/activate
```

### Issue: "Port 8000 already in use"
**Solution:** Kill existing process or use different port
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app:app --port 8001
```

### Issue: "CUDA out of memory" (shouldn't happen on CPU)
**Solution:** System is trying to use GPU. Force CPU:
```python
# In models.py, ensure:
self.device = "cpu"  # Should already be set
```

### Issue: Models download slowly
**Solution:** Models are cached after first download. If interrupted:
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/
python app.py  # Will re-download
```

### Issue: Frontend can't connect to backend
**Solution:** Check CORS and backend status
1. Ensure backend is running: `curl http://localhost:8000/health`
2. Check browser console for errors
3. Try opening frontend via http server instead of file://

---

## Performance Optimization

### Reduce Memory Usage

In `backend/models.py`:
```python
# Reduce batch size
embeddings = self.embedding_model.encode(
    texts,
    batch_size=4  # Reduce from 8 to 4
)
```

### Speed Up Inference

1. **Use smaller corpus:** Fewer passages = faster retrieval
2. **Reduce top-k:** Retrieve fewer evidence passages
3. **Shorter max_length:** Generate shorter answers

```python
# In app.py
evidence = retrieval_system.retrieve(question, top_k=3)  # Instead of 5
answer = model_manager.generate_answer(question, max_length=64)  # Instead of 128
```

### Preload Models

For repeated use, keep server running instead of restarting.

---

## Research Workflow

### 1. Data Collection
- Gather domain-specific corpus
- Create evaluation QA pairs with ground truth

### 2. System Configuration
- Adjust retrieval parameters
- Tune detection threshold on validation set

### 3. Run Evaluation
```bash
python run_evaluation.py
```

### 4. Analyze Results
- Check `results/metrics.json` for overall performance
- Review `results/evaluation_results.csv` for per-question analysis
- Examine visualizations

### 5. Iterate
- Adjust hyperparameters
- Expand corpus
- Re-evaluate

### 6. Publication
- Fill in research paper template with results
- Include visualizations from `results/`
- Report all hyperparameters for reproducibility

---

## Jupyter Notebook Integration

Create `analysis.ipynb` in backend/:

```python
# Cell 1: Setup
import sys
sys.path.append('.')
from models import get_model_manager
from retrieval import RetrievalSystem, create_sample_corpus
from hallucination import HallucinationDetector

# Cell 2: Load components
manager = get_model_manager()
retrieval = RetrievalSystem(manager.embedding_model)
corpus = create_sample_corpus("data/corpus.txt")
retrieval.build_index(corpus)
detector = HallucinationDetector(manager.embedding_model)

# Cell 3: Interactive testing
question = "What is the Eiffel Tower?"
answer = manager.generate_answer(question)
evidence = retrieval.retrieve(question, top_k=5)
result = detector.detect_hallucination(answer, evidence)

print(f"Answer: {answer}")
print(f"Hallucinated: {result['is_hallucinated']}")
print(f"Score: {result['hallucination_score']:.3f}")

# Cell 4: Visualization
import matplotlib.pyplot as plt
scores = [r['score'] for r in evidence]
plt.bar(range(len(scores)), scores)
plt.title("Evidence Relevance Scores")
plt.xlabel("Evidence Passage")
plt.ylabel("Relevance Score")
plt.show()
```

---

## Deployment

### Local Production

Use Gunicorn for production:
```bash
pip install gunicorn
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

### Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t hallucination-detection .
docker run -p 8000:8000 hallucination-detection
```

---

## Citation Format

For academic papers:

**APA:**
```
[Your Name]. (2024). Lightweight Hallucination Detection and Reduction 
in Local LLM Systems using Retrieval-Based Verification. 
[Unpublished research project].
```

**BibTeX:**
```bibtex
@misc{yourname2024hallucination,
  title={Lightweight Hallucination Detection and Reduction in Local LLM Systems},
  author={Your Name},
  year={2024},
  note={Research Project}
}
```

---

## Additional Resources

- **T5 Model:** https://huggingface.co/google/flan-t5-small
- **Sentence Transformers:** https://www.sbert.net/
- **FAISS Documentation:** https://github.com/facebookresearch/faiss
- **FastAPI Tutorial:** https://fastapi.tiangolo.com/tutorial/

---

**Happy Researching! 🔬**
