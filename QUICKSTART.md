# 🚀 QUICK START GUIDE

## Get Running in 5 Minutes

### Step 1: Setup (First Time Only)
```bash
cd "Hallucination Detection"
chmod +x setup.sh
./setup.sh
```

Wait for models to download (~400MB, 5-10 minutes).

### Step 2: Start the System
```bash
cd backend
source venv/bin/activate
python app.py
```

You should see:
```
✓ System ready!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Open the Demo
Open `frontend/index.html` in your browser.

### Step 4: Try It Out!
1. Enter a question (or click a sample question)
2. Click "Analyze Question"
3. View results:
   - Hallucination score
   - Original vs corrected answer
   - Evidence snippets
   - Claim-by-claim analysis

---

## Example Questions to Try

✅ **Factual (Should Pass):**
- "What is the capital of France?"
- "Who created Python programming language?"
- "What percentage of Earth is water?"

⚠️ **May Trigger Detection:**
- "When was the Eiffel Tower destroyed?"
- "How many legs does a spider have?"
- "What is the speed of light in kilometers?"

---

## Running Evaluation

```bash
cd backend
source venv/bin/activate
python run_evaluation.py
```

Check `results/` folder for:
- `metrics.json` - Overall performance
- `evaluation_results.csv` - Detailed results
- `*.png` - Visualizations
- `results_table.tex` - LaTeX table

---

## Project Structure at a Glance

```
📁 Hallucination Detection/
├── 📄 README.md              ← Full documentation
├── 📄 USAGE_GUIDE.md         ← Detailed usage examples
├── 🔧 setup.sh               ← One-click setup
│
├── 📁 backend/               ← Python system
│   ├── app.py                ← FastAPI server (START HERE)
│   ├── models.py             ← Model loading
│   ├── retrieval.py          ← FAISS search
│   ├── hallucination.py      ← Detection algorithm
│   ├── correction.py         ← Answer correction
│   ├── evaluation.py         ← Metrics & plots
│   ├── run_evaluation.py     ← Evaluation script
│   ├── requirements.txt      ← Dependencies
│   └── data/
│       ├── corpus.txt        ← Evidence database
│       └── qa_dataset.jsonl  ← Evaluation data
│
├── 📁 frontend/              ← Web interface
│   ├── index.html            ← Demo UI
│   ├── styles.css            ← Styling
│   └── script.js             ← API calls
│
├── 📁 results/               ← Generated outputs
│   └── (created after evaluation)
│
└── 📁 paper/                 ← Research template
    └── research_paper_template.md
```

---

## Key Features

✅ **Small Models** - Runs on 8GB RAM, no GPU  
✅ **Fast** - <5 second response time  
✅ **Interpretable** - Claim-level scoring  
✅ **Complete** - API + Frontend + Evaluation  
✅ **Reproducible** - All code documented  

---

## Models Used

| Component | Model | Size | Speed |
|-----------|-------|------|-------|
| LLM | flan-t5-small | 77M params | Fast |
| Embeddings | all-MiniLM-L6-v2 | 22M params | Very Fast |

---

## Troubleshooting

**❌ "Port 8000 already in use"**
```bash
lsof -ti:8000 | xargs kill -9
```

**❌ "Module not found"**
```bash
cd backend
source venv/bin/activate  # Activate venv first!
```

**❌ Frontend can't connect**
- Ensure backend is running (`curl http://localhost:8000/health`)
- Check browser console for errors

**❌ Slow first run**
- Normal! Models download on first run
- Subsequent runs are much faster

---

## Next Steps

1. ✅ **Run the demo** - Try sample questions
2. 📊 **Run evaluation** - Generate metrics
3. 📝 **Read paper template** - Understand methodology
4. 🔬 **Customize** - Add your own corpus data
5. 📄 **Publish** - Use for research/coursework

---

## API Testing (Optional)

Test without frontend:

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?"}' | python -m json.tool
```

---

## Resources

- 📖 Full docs: `README.md`
- 📚 Usage guide: `USAGE_GUIDE.md`
- 📄 Paper template: `paper/research_paper_template.md`

---

## Support

Issues? Check:
1. Virtual environment is activated
2. Backend server is running
3. Models finished downloading
4. Port 8000 is available

---

**Ready to detect hallucinations! 🎯**
