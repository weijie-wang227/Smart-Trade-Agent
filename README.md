# Smart Trade Agent — Setup & Model Choices

## Setup & Installation

This project can be run either locally using a Python virtual environment or via Docker for a fully containerized setup.

### Option A — Local Setup (Python venv)

1. Clone the repository
```
git clone <your-repo-url>
cd smart-trade-agent
```

2. Create and activate a virtual environment
```
python -m venv venv
```

Windows (PowerShell):
```
.\venv\Scripts\Activate.ps1
```

macOS / Linux:
```
source venv/bin/activate
```

3. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

4. Set up environment variables

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Run the FastAPI server
```
uvicorn app.main:app --reload
```

You should see logs similar to:
```
🚀 Lifespan startup begin
📦 Vector store count: 10
✅ Lifespan startup complete
Uvicorn running on http://127.0.0.1:8000
```

6. Open the API docs

Navigate to:
```
http://127.0.0.1:8000/docs
```

From here you can test:

- POST /suggest
- POST /suggest_hybrid
- GET /benchmark
- GET /benchmark_hybrid

---

### Option B — Docker Setup (Recommended for Reproducibility)

1. Build the Docker image
```
docker build -t smart-trade-agent .
```

2. Create a `.env` file
```
GEMINI_API_KEY=your_gemini_api_key_here
```

3. Run using Docker Compose
```
docker compose up
```

(or for older Docker versions)
```
docker-compose up
```

4. Open the API docs
```
http://localhost:8000/docs
```

---

## Model Choices & Anomaly Handling

### 1. LLM Choice — Gemini 2.0 Flash

This system uses Google Gemini 2.0 Flash as the optional verification LLM in the final stage of the classification pipeline.

**Rationale**

- **Lowest-cost Gemini tier**  
  Gemini 2.0 Flash offers the cheapest token pricing among Gemini models, making it suitable for frequent, lightweight validation calls.

- **Low latency**  
  The model is optimized for fast responses, which helps keep end-to-end inference time well below the 60-second requirement.

- **Sufficient reasoning capability**  
  Although it is a lightweight model, Gemini 2.0 Flash is able to:
  - Compare retrieved HS-code candidates
  - Reason over product descriptions
  - Assign a plausibility score for borderline cases

  This is sufficient for validation and confidence calibration tasks.

- **Cost-efficient design**  
  The LLM is only invoked for borderline-confidence cases (confidence between 0.5 and 0.8).  
  Most queries incur zero LLM cost.

**Typical verification cost**

- Prompt tokens: approximately 180–220  
- Output tokens: approximately 1–5  
- Cost per verification: approximately $0.00002–$0.00007 USD

This design keeps the amortized cost per query extremely low.

---

### 2. Embedding Model Choice — all-MiniLM-L6-v2

The vector store uses the HuggingFace embedding model:

```
sentence-transformers/all-MiniLM-L6-v2
```

**Rationale**

- **Strong semantic retrieval quality**  
  all-MiniLM-L6-v2 performs well for short technical and product-style text.

- **Lightweight and fast**  
  - Approximately 22M parameters  
  - Very fast CPU inference  
  - Suitable for local and containerized deployment

- **Well-established industry baseline**  
  This model is widely used in RAG pipelines and provides a strong trade-off between:
  - Retrieval accuracy
  - Latency
  - Memory usage

- **Offline / local friendly**  
  No external API calls are required for embeddings, improving reliability and reproducibility.

---

### 3. Handling the “Anomaly” Test Case

**Wireless Bluetooth headphones with integrated solar charging panels**

This test case is intentionally ambiguous:

Should the product be classified under:
- Electronics (8517 / 8518)

or

- Solar technology (8541)?

A good agent should flag the overlap and ask for clarification rather than confidently selecting one category.

---

#### 3.1 Multi-signal ambiguity detection

The system does not blindly trust vector similarity. Instead, it uses multiple independent signals:

- Dense semantic similarity (Chroma + MiniLM)
- Category disagreement among top retrieved candidates
- Keyword-based domain flags

Examples:

- `electronics` detected from: bluetooth, headphones, wireless
- `renewables` detected from: solar, panel

If the top retrieved candidates:

- Belong to different HS categories
- Have high semantic similarity scores
- And the input text activates multiple domain flags

Then the system marks the case as semantically conflicting.

---

#### 3.2 Confidence gating & manual review escalation

Even if one category appears dominant, the system applies a hard ambiguity gate:

- If cross-domain conflict is detected:  
  → confidence is capped  
  → manual review is triggered

This prevents over-confidence for composite or hybrid products.

---

#### 3.3 Controlled use of the LLM verifier

For borderline-confidence cases, Gemini is used to estimate relative plausibility between candidates.

However, the LLM is not allowed to override ambiguity signals.

If a conflict is detected:

- The LLM score is capped
- The final confidence cannot exceed a safe threshold
- The case is escalated to manual review

This ensures that:

- The system prefers safe abstention over confident misclassification.

---

### 4. Summary of Design Philosophy

| Component              | Design Goal                                |
|------------------------|---------------------------------------------|
| MiniLM embeddings      | Fast, local, strong semantic retrieval      |
| Chroma vector store    | Efficient similarity search                 |
| Gemini 2.0 Flash       | Low-cost, low-latency validation            |
| Confidence gating      | Prevent false positives                     |
| Anomaly detection      | Escalate cross-domain products              |

