## 🎲 Board Game Rulebook Q&A Chatbot (RAG) - CATAN Game Master

> **Ask any question about CATAN. Get instant, rule-accurate answers grounded in the official manual.**

Have a question about how to play **CATAN**?<br>
Just type your question into the input box and press **Enter** (or click **Search**).<br>
You'll instantly see:
- **Your question and the generated answer**, shown below the input
- An expandable **“View Source Passages”** section, where you can read the top-K rulebook snippets the system used to generate the answer

Not sure what to ask? <br>
Explore the **"Popular Questions"** section and click any question to auto-fill it and generate an answer.

---

### **Live Demo** → **[Link](https://broadgame-question-and-answer.streamlit.app/)** (Mobile & desktop friendly)

![til](https://github.com/khchu93/NoteImage/blob/main/board_game_demo.gif)

---

### What This Project Demonstrates
- **End-to-end RAG system** with chunking, embedding, vector search, and LLM answering  
- **High-fidelity answers** grounded strictly in top-k retrieved rule text, with prompt constraints to minimize hallucination.
- **Full retrieval evaluation:** NDCG, MRR, Mean Retrieval Similarity(MRS), HitRate@K  
- **Full generation evaluation:** RAGAS (Faithfulness, Relevancy, Correctness)  
- **Hyperparameter tuning** across chunk size, overlap, k, and similarity metrics  
- **Clean, mobile-friendly Streamlit UI**  
- **Modular, production-style architecture** (document loader, retriever, evaluator, UI)

---

### Key Results

#### **Best Retrieval Settings**<br>
*Selected through retrieval evaluation across multiple chunk sizes, overlaps, and top-k values, using a 10-question ground-truth Q&A dataset derived from the CATAN rulebook.*
- **Chunk Size:** 125  
- **Chunk Overlap:** 120  
- **Top-K:** 5  
- **Similarity search:** Cosine  <br>
These achieved the highest average retrieval score across [nDCG](#Retrieval-Metrics), [MRR](#Retrieval-Metrics), [Overall-MRS](#Retrieval-Metrics), and [Mean HitRate@K](#Retrieval-Metrics).

| Summary Metric | Chunk | Overlap | Top-K | Score |
|----------------|--------|----------|--------|--------|
| [Average](#Retrieval-Metrics) | 125 | 120 | 5 | **0.7874** |
| [Prioritize Ranking](#Retrieval-Metrics) | 125 | 120 | 5 | **0.7647** |

#### **Generation Evaluation (RAGAS)**
*RAGAS scores computed using a 40-question custom Q&A dataset grounded in the official CATAN rulebook, designed to evaluate answer faithfulness, relevancy, correctness, and context quality.*

| RAGAS Metric | Mean | Std |
|--------|--------|--|
| [Faithfulness](#Summary-Metrics) | 0.6958 | 0.4046 |
| [Answer Relevancy](#Summary-Metrics) | 0.9694 | 0.0389 |
| [Answer Correctness](#Summary-Metrics) | 0.6163 | 0.2121 |
| [Context Precision](#Summary-Metrics) | 0.7921 | 0.2983 |
| [Context Recall](#Summary-Metrics) | 0.8750 | 0.3307 |

---

### Why I Built This
When learning a new board game, I constantly interrupt gameplay to flip through rulebooks and verify details—breaking the flow and killing the fun.

I built this chatbot to act as a personal game master:
- instantly answers rule questions
- stays grounded in the official rulebook
- removes all manual page-flipping during gameplay
- allows players to focus on fun, not rules

---

### **How It Works**

```mermaid
graph LR
    A[Q&A Dataset] --> B[Embeddings]
    B --> C[Vector Search]
    C --> D[Top-K Retrieval]
    E[Prompt Template] --> F[LLM Generation]
    L[Question] --> F[LLM Generation]
    D --> F[LLM Generation]
    F --> G[Answer + Sources]
    
    H[PDF Manual] --> I[Chunking]
    I --> J[Embeddings]
    J --> K[ChromaDB]
    K --> C
```

---

### **Technology Stack**

<table>
<tr>
<td width="50%">

**LLM**
- GPT-3.5 Turbo (generation, chatbot, ragas evaluation)
- text-embedding-ada-002 (embeddings)
- LangChain framework

</td>
<td width="50%">

**Vector Database**
- ChromaDB (Vector Store)
- Cosine similarity search

</td>
</tr>
<tr>
<td>

**Evaluation**
- RAGAS (Generation Metrics)
- DCG/nDCG/MRR/Mean MRS/Mean HitRate@K (Retrieval Metrics)

</td>
<td>

**Interface**
- Streamlit (web app & Mobile)

</td>
</tr>
</table>

---

### **Design Decisions**

<table>
<tr>
<td width="50%">

**Chunking (125 size / 120 overlap / top-k=5)**

Board game manuals contain short, dense rule sentences.<br>
This configuration maximized retrieval accuracy because:<br>
- high overlap preserves rule integrity
- moderate chunk size avoids fragmented meaning
- small k reduces irrelevant noise

</td>
<td width="50%">

**Cosine similarity**

Compared cosine, L2, inner product — all similar results.<br>
Kept cosine because:<br>
- widely used in embedding-based retrieval
- stable and predictable
- excellent support in ChromaDB

</td>
</tr>
<tr>
<td>

**GPT-3.5-Turbo**

Chosen for:<br>
- low cost during hyperparameter search
- deterministic behavior at temperature 0
- high enough quality for constrained, rule-grounded answers

</td>
<td>

**ChromaDB**

Chosen because it is:<br>
- fast and lightweight
- persistent without needing a server
- integrates smoothly with LangChain
- ideal for experiment-heavy RAG workflows

</td>
</tr>
</table>

---


### Metric Detail
#### Retrieval Metrics

| Metric | Description | Scale | Interpretation |
|:--------|:----------------|:-----|:---|
| **Average NDCG(Normalized Discounted Cumulative Gain)** | Measures how close the retrieved ranking is to an ideal ordering. | 0 - 1 | Shows how close the chunk ranking is to the ideal |
| **MRR(Mean Reciprocal Rank)** | Measures how early the first relevant chunk appears. | 0 - 1 | Shows how early a relevant chunk is retrieved |
| **Overall MRS(Mean Retrieval Similarity)** | Measures the average semantic similarity between queries and top-k retrieved chunks | 0 - 1 | Shows how semantically close retrieved chunks are to the query. |
| **Mean HitRate@K** | Measures how often top-k contains at least one relevant chunk. | 0 - 1 | Shows whether top-k consistently includes a relevant chunk |

#### Summary Metrics
| Summary Metric | Equation |
|:--------|:----------------|
| **Average** | 0.2 x Average NDCG + 0.2 x MRR + 0.2 x Overall MRS + 0.2 x Mean HitRate@K |
| **Ranking Prioritize** | 0.4 x Average NDCG + 0.4 x MRR + 0.1 x Overall MRS + 0.1 x Mean HitRate@K |

[Back](#Key-Results)
#### Generation Metrics

| RAGAS Metric | Description | Scale | Interpretation |
|:--------|:-------------------------------|:------|:-------------------------------|
| **Faithfulness** | Measures whether the generated answer is consistent with the retrieved context. | 0–1 | Shows how closely the answer follows the source context without hallucination |
| **Answer Relevancy** | Measures how relevant the generated answer is to the user’s question. | 0–1 | Shows how well the answer addresses the question |
| **Answer Correctness** | Measures whether the generated answer is factually correct based on the reference. | 0–1 | Shows whether the answer contains correct information |
| **Context Precision** | Measures the fraction of retrieved chunks that are actually relevant. | 0–1 | Shows how many retrieved chunks are truly relevant to the query |
| **Context Recall** | Measures the fraction of all relevant chunks that were successfully retrieved. | 0–1 | Shows how completely relevant chunks are captured by the retriever |

[Back](#Key-Results)

---

### **Evaluation Results**

#### **Retrieval Evaluation**
> 📈 **Retrieval Evaluation (10-question annotated dataset)**
<img src="https://github.com/khchu93/NoteImage/blob/main/board_game_eval_heatmap.png" width="900" />
<img src="https://github.com/khchu93/NoteImage/blob/main/board_game_eval_all.PNG" width="900" />

|  | Chunk Size | Chunk Overlap | Top k |
|--|--|--|--|
|Best Combination: | 125 | 120 | 5 |
 
#### **Generation Evaluation (RAGAS)**
> 📈 **Generation Evaluation (RAGAS, 40-question dataset using optimal retrieval settings)**

|  | Faithfulness | Answer Relevancy | Answer Correctness | Context Precision | Context Recall |
|--|--|--|--|--|--|
|Score: | 0.7583 | 0.9773 | 0.6354 | 0.95 | 0.95 |

---

### Current Limitations
1. **Only supports one rulebook at a time**<br>
    The Streamlit UI defaults to CATAN; no multi-manual support yet.
2. **Q&A dataset has mostly 1 relevant chunk per question**<br>
    Board game rules tend to be concise single-sentence statements.
3. **Retrieval sometimes returns 0 relevant chunks**<br>
    Likely due to:
    - chunk boundaries
    - embedding limitations
    - semantic overlap issues
4. **RAGAS debugging is difficult**<br>
    When evaluating multiple parameter combinations, some rows in
    `rag_generation_eval.csv` becomes blank due to:
    - token limits
    - silent failures
    - scoring edge cases
5. **UI lacks feature toggles**<br>
    No user-adjustable chunking settings or model options (planned).

---

### **Project Structure**

```
rag-board-game-qa/
├─── demo.py                       # Demo
├─── rag_experiments.ipynb         # Analysis & visualization
├── .streamlit/
│   ├── config.toml                # Web UI setup
├── app/
│   ├── streamlit_app.py           # Web interface
├── src/
│   ├── __init__.py                # Version control
│   ├── rag_system.py              # Core RAG pipeline
│   ├── document_loader.py         # PDF processing
│   ├── chunking.py                # Text chunking strategies
│   ├── vector_store.py            # ChromaDB operations
│   ├── annotation.py              # Ground truth annotation
│   ├── exception.py               # Exception management
│   ├── prompts.py                 # Prompts templates management
│   └── config.py                  # Configuration management
├── evaluation/
│   ├── __init__.py                # Version control
│   ├── evaluation.py              # Evaluation pipeline
│   ├── metrics.py                 # DCG/nDCG implementation
│   ├── run_evaluation.py          # Evaluation runner
│   └── results_csv/               # CSV outputs
├── data/
    └── BoardGamesRuleBook/        # Game manuals & test data
```

---

### **Quick Start**

#### **1. Live Demo**
Visit **[Link](https://rag-board-game.streamlit.app/)**

#### **2. Run Locally**

```bash
# Clone the repository
git clone https://github.com/khchu93/Do-my-history-exam.git
cd Do-my-history-exam

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_key_here" > .env

# Run Streamlit app
streamlit run app/streamlit_app.py

# OR run CLI demo
python app/demo.py
```

---

### **Run Your Own Evaluation**

```bash
# Run full evaluation pipeline
python evaluation/run_evaluation.py

# Results saved to:
# - evaluation/results/retrieval_eval.csv
# - evaluation/results/generation_eval.csv
```

This will:
- Test different chunking strategies (chunk_size, chunk_overlap, k)
- Evaluate retrieval quality (DCG/nDCG/MRR/Overall MRS/Mean HitRate@K)
- Measure generation quality (RAGAS)
- Export results for analysis

---

### **Configuration**

Hyperparameters in `src/config.py`:

```python
# Paths
DATA_DIR = Path("data/BoardGamesRuleBook")
PDF_PATH = DATA_DIR / "CATAN.pdf"
# TRAINING_QA_PATH = DATA_DIR / "CATAN_eval_small.json" # retrieval eval set
TRAINING_QA_PATH = DATA_DIR / "CATAN_eval_long.json"    # generation eval set

# Model settings
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0
SIMILARITY_SEARCH = "cosine"

# Chunking parameters (demo.py and streamlit_app.py)
DEMO_CHUNK_SIZE = 125
DEMO_CHUNK_OVERLAP = 120
DEMO_TOP_K = 5

# Evaluation parameters (run_evaluation.py)
CHUNK_SIZES = [125]
CHUNK_OVERLAPS = [120]
TOP_K_VALUES = [5]

# Prompt template
PROMPT_TEMPLATE = "default"
```

---

### **What I Learned**
- Designing full RAG pipelines
- Retrieval + generation evaluation methodology
- Chunking and embedding optimization
- Prompt engineering for rule-faithful answers
- Building modular, production-style Python systems
- Creating reliable custom evaluation datasets
- Using RAGAS for grounded LLM evaluation
- Visualization and experiment tracking

---

### **Future Enhancements**

- [ ] Multi-rulebook support
- [ ] Rulebook upload in UI
- [ ] OCR for photographed manuals
- [ ] Multilingual rulebook support
- [ ] Query rewriting / semantic expansion
- [ ] More advanced chunking strategies
- [ ] More advanced prompt template strategies

---
