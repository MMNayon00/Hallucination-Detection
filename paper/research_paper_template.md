# Lightweight Hallucination Detection and Reduction in Local LLM Systems using Retrieval-Based Verification

**Research Paper Template**

---

## Abstract

**Context:** Large Language Models (LLMs) frequently generate plausible-sounding but factually incorrect information, known as hallucinations. This poses significant challenges for deploying LLMs in resource-constrained environments.

**Problem:** Existing hallucination detection methods often require substantial computational resources, making them unsuitable for local deployment on consumer-grade hardware.

**Solution:** We propose a lightweight hallucination detection and reduction system that combines retrieval-based verification with semantic similarity scoring, optimized for CPU-only inference on devices with limited memory (8GB RAM).

**Method:** Our approach uses small-scale models (google/flan-t5-small for generation, sentence-transformers/all-MiniLM-L6-v2 for embeddings) with FAISS-based retrieval to verify generated claims against a local corpus. We implement claim-level verification scoring and evidence-grounded answer correction.

**Results:** [TO BE FILLED AFTER EXPERIMENTS]
- Hallucination detection accuracy: X%
- Answer quality improvement: Y%
- System operates efficiently on MacBook Air M2 with 8GB RAM
- Average inference time: Z seconds per question

**Conclusion:** Our system demonstrates that effective hallucination detection and reduction can be achieved without requiring large-scale computational resources, making it suitable for academic research and resource-constrained deployments.

**Keywords:** Hallucination Detection, Large Language Models, Retrieval-Augmented Generation, Resource-Efficient AI, Natural Language Processing

---

## 1. Introduction

### 1.1 Motivation

Large Language Models have revolutionized natural language processing tasks, but their tendency to generate hallucinated content remains a critical challenge. While cloud-based solutions exist, many researchers and practitioners require local, privacy-preserving systems that can run on consumer hardware.

### 1.2 Research Questions

1. **RQ1:** Can hallucinations be effectively detected using lightweight models and retrieval-based verification?
2. **RQ2:** Does claim-level semantic similarity scoring provide reliable hallucination detection?
3. **RQ3:** Can evidence-grounded prompting reduce hallucination rates in small language models?
4. **RQ4:** What is the trade-off between model size and detection accuracy in resource-constrained environments?

### 1.3 Contributions

This work makes the following contributions:

1. **Lightweight Architecture:** A complete hallucination detection system optimized for CPU-only inference on 8GB RAM devices
2. **Claim-Level Verification:** A novel scoring mechanism based on semantic similarity between claims and retrieved evidence
3. **Evidence-Grounded Correction:** An automatic answer correction pipeline using retrieval-augmented generation
4. **Reproducible Evaluation:** Comprehensive evaluation pipeline with publication-ready metrics and visualizations
5. **Open Implementation:** Fully documented, reproducible codebase for academic research

---

## 2. Related Work

### 2.1 Hallucination in Language Models

- **Factual Errors:** [Survey papers on hallucination types]
- **Detection Methods:** [Previous work on hallucination detection]
- **Metrics:** [Common evaluation approaches]

### 2.2 Retrieval-Augmented Generation

- **RAG Architectures:** [DPR, REALM, RAG paper citations]
- **Verification Methods:** [Fact-checking with retrieval]
- **Small-Scale Systems:** [Work on efficient RAG]

### 2.3 Resource-Efficient LLMs

- **Model Compression:** [Distillation, quantization approaches]
- **Small Models:** [T5-small, DistilBERT performance studies]
- **Efficient Inference:** [CPU optimization techniques]

**Gap in Literature:** Existing hallucination detection systems primarily target large-scale models with GPU access. Our work addresses the under-explored area of lightweight, CPU-friendly hallucination detection.

---

## 3. Methodology

### 3.1 System Architecture

Our system consists of four main components:

1. **Answer Generation Module**
   - Model: google/flan-t5-small (77M parameters)
   - CPU-optimized inference
   - Deterministic decoding for reproducibility

2. **Retrieval Module**
   - Embedding: sentence-transformers/all-MiniLM-L6-v2 (22M parameters)
   - Vector store: FAISS with L2 distance
   - Top-k retrieval (k=5)

3. **Hallucination Detection Module**
   - Claim extraction via sentence segmentation
   - Semantic similarity computation
   - Aggregated hallucination scoring

4. **Answer Correction Module**
   - Evidence-grounded prompt engineering
   - Constrained generation from retrieved sources

### 3.2 Hallucination Detection Algorithm

**Input:** Generated answer A, retrieved evidence E = {e₁, e₂, ..., eₖ}

**Algorithm:**

```
1. Extract claims C = {c₁, c₂, ..., cₙ} from A using sentence segmentation
2. For each claim cᵢ:
   a. Compute embedding vectors: emb(cᵢ), {emb(eⱼ) | eⱼ ∈ E}
   b. Calculate similarity: sim(cᵢ, eⱼ) = cosine(emb(cᵢ), emb(eⱼ))
   c. Claim score: s(cᵢ) = max{sim(cᵢ, eⱼ) | eⱼ ∈ E}
3. Aggregate hallucination score: H = 1 - mean({s(cᵢ)})
4. Classification: hallucinated = (H > τ), where τ = 0.45
```

**Rationale:**
- Claims well-supported by evidence have high similarity scores
- Hallucinated claims have low maximum similarity to any evidence
- Inverse mean provides intuitive hallucination probability

### 3.3 Evidence-Grounded Correction

When hallucination is detected (H > τ):

```
Prompt = "Based on the following evidence, answer the question accurately.
Only use information from the evidence provided.

Evidence:
[Top-3 retrieved passages]

Question: [Original question]

Answer:"
```

This constrains the model to generate answers grounded in verifiable evidence.

---

## 4. Experimental Setup

### 4.1 Hardware and Software

- **Hardware:** MacBook Air M2, 8GB RAM, 512GB SSD
- **Software:** Python 3.10, PyTorch 2.1.0, FastAPI 0.104.1
- **Models:**
  - LLM: google/flan-t5-small
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2
- **Inference:** CPU-only (no GPU required)

### 4.2 Dataset

**Corpus:**
- Source: [Wikipedia excerpts / Custom knowledge base]
- Size: [Number of documents/passages]
- Processing: Sentence-level chunking with 50-character overlap

**Evaluation Dataset:**
- Format: Question-answer pairs with ground truth
- Size: [Number of QA pairs]
- Source: [TruthfulQA subset / Custom dataset]
- Annotation: Binary hallucination labels where available

### 4.3 Evaluation Metrics

**Hallucination Detection (if labels available):**
- Precision, Recall, F1-score
- Accuracy
- ROC-AUC

**Answer Quality:**
- Semantic similarity to ground truth (cosine similarity)
- Accuracy before correction
- Accuracy after correction
- Improvement rate

**Efficiency:**
- Inference time per question
- Memory usage
- Model loading time

### 4.4 Baselines

1. **No Detection:** Direct LLM generation without verification
2. **Random Detection:** Random hallucination classification
3. **Confidence-Based:** Using model confidence scores (if available)

### 4.5 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Retrieval k | 5 | Balance between context and noise |
| Detection threshold τ | 0.45 | Optimized on validation set |
| Max answer length | 128 tokens | Concise, factual answers |
| Chunk size | 200 characters | Granular evidence retrieval |

---

## 5. Results

### 5.1 Hallucination Detection Performance

**Table 1: Detection Metrics**

| Metric | Value |
|--------|-------|
| Precision | [X%] |
| Recall | [Y%] |
| F1-Score | [Z%] |
| Accuracy | [W%] |
| ROC-AUC | [V] |

**Figure 1:** ROC Curve for hallucination detection
*[results/roc_curve.png]*

**Figure 2:** Distribution of hallucination scores
*[results/score_distribution.png]*

### 5.2 Answer Quality Improvement

**Table 2: Answer Quality Metrics**

| Condition | Avg. Similarity | Improvement |
|-----------|----------------|-------------|
| Before Correction | [X] | - |
| After Correction | [Y] | [+Z%] |

**Figure 3:** Accuracy comparison before/after correction
*[results/accuracy_comparison.png]*

### 5.3 Efficiency Analysis

**Table 3: Performance Characteristics**

| Metric | Value |
|--------|-------|
| Model Loading Time | [X] seconds |
| Avg. Inference Time | [Y] seconds |
| Peak Memory Usage | [Z] GB |
| Throughput | [W] questions/minute |

### 5.4 Qualitative Analysis

**Example 1: Successful Detection**
```
Question: [...]
Generated Answer: [...]
Hallucination Score: 0.72 (DETECTED)
Corrected Answer: [...]
Outcome: Correction improved factual accuracy
```

**Example 2: False Positive**
```
[Analysis of system limitations]
```

---

## 6. Discussion

### 6.1 Key Findings

1. **RQ1:** Retrieval-based verification achieves [X%] detection accuracy with lightweight models
2. **RQ2:** Claim-level scoring provides interpretable, granular hallucination assessment
3. **RQ3:** Evidence-grounded correction improves answer quality by [Y%]
4. **RQ4:** Trade-off: [Discussion of size vs. performance]

### 6.2 Advantages of the Approach

- **Accessibility:** Runs on consumer hardware without GPU
- **Interpretability:** Claim-level scores explain detection decisions
- **Privacy:** Complete local processing, no cloud dependencies
- **Modularity:** Components can be upgraded independently

### 6.3 Limitations

1. **Corpus Dependency:** Detection quality depends on corpus coverage
2. **Model Size:** Small models may miss complex hallucinations
3. **Latency:** Retrieval adds computational overhead
4. **Language Support:** Currently optimized for English

### 6.4 Comparison with Large-Scale Systems

[Qualitative comparison with GPT-4, Claude, etc. - acknowledging different resource requirements]

---

## 7. Ablation Studies

### 7.1 Component Analysis

**Table 4: Ablation Results**

| Configuration | Detection F1 | Answer Quality |
|---------------|-------------|----------------|
| Full System | [X] | [Y] |
| w/o Retrieval | [X-a] | [Y-b] |
| w/o Correction | [X] | [Y-c] |
| Random Baseline | [X-d] | [Y] |

### 7.2 Threshold Sensitivity

[Analysis of detection threshold τ impact]

### 7.3 Retrieval Size Impact

[Effect of varying k in top-k retrieval]

---

## 8. Ethical Considerations

### 8.1 Responsible AI

- System reduces but does not eliminate hallucinations
- Users should verify critical information
- Intended for research and educational purposes

### 8.2 Bias and Fairness

- Corpus biases affect detection and correction
- Small models may underperform on diverse topics
- Requires diverse, representative knowledge bases

### 8.3 Environmental Impact

- CPU-only inference reduces energy consumption
- Smaller models have lower carbon footprint
- Encourages sustainable AI research

---

## 9. Future Work

1. **Multilingual Support:** Extend to non-English languages
2. **Active Learning:** Automatically expand corpus with verified facts
3. **Model Distillation:** Further compress models while maintaining accuracy
4. **Real-time Fact-Checking:** Integration with live knowledge sources
5. **Uncertainty Quantification:** Probabilistic hallucination estimates
6. **Domain Adaptation:** Specialized systems for medical, legal domains

---

## 10. Conclusion

We presented a lightweight hallucination detection and reduction system that achieves [X%] detection accuracy and [Y%] answer quality improvement using only CPU resources on 8GB RAM devices. Our claim-level verification approach provides interpretable hallucination assessment, and evidence-grounded correction demonstrably improves factual accuracy.

This work demonstrates that effective hallucination mitigation is achievable without large-scale computational resources, making it accessible for academic research, education, and privacy-sensitive applications. The complete system is open-sourced to facilitate reproducible research in resource-efficient NLP.

---

## Acknowledgments

[Funding sources, collaborators, computational resources]

---

## References

[To be filled with proper citations]

1. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5 paper)
2. Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
3. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
4. [TruthfulQA paper]
5. [Hallucination survey papers]
6. [FAISS paper]
7. [Additional relevant citations]

---

## Appendix A: Implementation Details

### A.1 Model Specifications

**google/flan-t5-small:**
- Parameters: 77M
- Architecture: Encoder-decoder transformer
- Context length: 512 tokens
- Download size: ~300MB

**sentence-transformers/all-MiniLM-L6-v2:**
- Parameters: 22M
- Embedding dimension: 384
- Download size: ~90MB

### A.2 API Endpoints

```
POST /ask
Input: {"question": "string"}
Output: {
  "answer": "string",
  "is_hallucinated": bool,
  "hallucination_score": float,
  "corrected_answer": "string | null",
  "evidence": [...]
}
```

### A.3 Reproducibility Checklist

- [x] Code publicly available
- [x] Dependencies specified (requirements.txt)
- [x] Hardware requirements documented
- [x] Random seeds fixed for deterministic results
- [x] Evaluation datasets provided
- [x] Model versions specified

---

## Appendix B: Experimental Results Tables

[Detailed tables with all experimental runs]

---

## Appendix C: Qualitative Examples

[Additional examples of system behavior across different question types]

---

*End of Research Paper Template*
