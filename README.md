# Board Game Manual Q&A System 🎲

A production-ready **Retrieval-Augmented Generation (RAG)** system for answering questions about board game manuals. This project demonstrates best practices in RAG system design, evaluation, and deployment.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Key Features

- **Production RAG System**: Clean, modular architecture ready for deployment
- **Comprehensive Evaluation**: Coverage-based metrics (DCG/nDCG) + RAGAS generation metrics
- **Ground Truth Integration**: Aho-Corasick pattern matching for efficient annotation
- **Interactive Demo**: User-friendly CLI for testing the system
- **Parameter Optimization**: Grid search over chunking and retrieval parameters

## 📁 Project Structure

```
.
├── config.py              # Configuration and settings
├── exceptions.py          # Custom exception classes
├── document_loader.py     # PDF loading and preprocessing
├── annotation.py          # Ground truth Q&A annotation
├── chunking.py            # Document chunking and coverage calculation
├── vector_store.py        # Vector store operations (Chroma)
├── metrics.py             # Evaluation metrics (DCG/nDCG)
├── rag_system.py          # Core RAG system (production)
├── evaluation.py          # Evaluation pipeline
├── demo.py               # Interactive demo
├── run_evaluation.py     # Evaluation runner
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-board-game-qa.git
cd rag-board-game-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Place your data files in the correct location:
```
data/
└── BoardGamesRuleBook/
    ├── CATAN.pdf
    └── CATAN_train_small.json
```

### 3. Run Interactive Demo

```bash
python demo.py
```

Example interaction:
```
❓ Your question: How do you win the game?

🔍 Searching for relevant information...

──────────────────────────────────────────────────────────────────────
💡 ANSWER:
──────────────────────────────────────────────────────────────────────
To win the game, you need to be the first player to reach 10 victory 
points. Victory points are earned through building settlements (1 point), 
cities (2 points), having the longest road (2 points), the largest army 
(2 points), and certain development cards.
──────────────────────────────────────────────────────────────────────
```

### 4. Run Evaluation

```bash
python run_evaluation.py
```

This will:
- Test different parameter configurations
- Evaluate retrieval quality (DCG/nDCG)
- Evaluate generation quality (RAGAS metrics)
- Save results to CSV files

## 📊 Evaluation Methodology

### Retrieval Evaluation

**Coverage-Based Relevance**:
- Uses ground truth Q&A annotations
- Calculates coverage: `overlap_length / relevance_span_length`
- Accounts for partial matches across chunk boundaries

**Metrics**:
- **DCG** (Discounted Cumulative Gain): Position-aware relevance scoring
- **nDCG** (Normalized DCG): Comparable across queries (0-1 scale)

### Generation Evaluation

**RAGAS Metrics**:
- **Answer Correctness**: Semantic similarity to ground truth
- **Answer Relevancy**: How well answer addresses the question
- **Faithfulness**: Whether answer is grounded in retrieved context
- **Context Precision**: Relevance of retrieved chunks to question
- **Context Recall**: Coverage of ground truth in retrieved context

## 🏗️ System Architecture

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Embedding Model │ (OpenAI Ada-002)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Search  │ (ChromaDB)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Top-K Chunks   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM (GPT-3.5) │ + Context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

## 🔧 Technical Highlights

### 1. Efficient Pattern Matching
Uses **Aho-Corasick algorithm** for ground truth annotation:
- Time complexity: O(n + m + z) vs O(n×m) for naive search
- Handles 100+ patterns efficiently
- Critical for large-scale evaluation

### 2. Coverage-Based Scoring
Novel approach to relevance measurement:
- Handles spans that cross chunk boundaries
- Provides granular scores (not just binary)
- Enables meaningful nDCG calculation

### 3. Modular Design
Clean separation of concerns:
- Easy to swap components (different embeddings, LLMs, vector stores)
- Testable individual modules
- Production-ready code structure

## 📈 Example Results

```
Retrieval Evaluation:
├── Average DCG:   0.4697
└── Average nDCG:  0.5262

Generation Evaluation:
├── Answer Correctness:  0.7145 ± 0.12
├── Answer Relevancy:    0.8234 ± 0.09
├── Faithfulness:        0.8567 ± 0.11
├── Context Precision:   0.6891 ± 0.15
└── Context Recall:      0.7423 ± 0.13
```

## 🎯 Use Cases

- **Customer Support**: Automated Q&A for product manuals
- **Education**: Interactive learning from textbooks
- **Legal/Compliance**: Quick reference for policy documents
- **Technical Documentation**: Developer Q&A systems

## 🛠️ Future Enhancements

- [ ] Add support for multi-document retrieval
- [ ] Implement hybrid search (dense + sparse)
- [ ] Add streaming responses
- [ ] Build web interface (Streamlit/Gradio)
- [ ] Add citation/source tracking
- [ ] Implement feedback loop for continuous improvement

## 📚 References

- **RAG + Langchain Tutorial**: [YouTube](https://www.youtube.com/watch?v=tcqEUSNCn8I)
- **RAGAS Framework**: [Docs](https://docs.ragas.io/)
- **LangChain**: [Docs](https://python.langchain.com/)

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for junior LLM engineer positions**