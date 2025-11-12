"""
Configuration settings for the RAG evaluation system.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except:
    # Fallback to .env file for local development
    load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
DATA_DIR = Path("data/BoardGamesRuleBook")
PDF_PATH = DATA_DIR / "CATAN.pdf"
TRAINING_QA_PATH = DATA_DIR / "CATAN_train_small.json"

# Model settings
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0

# Chunking parameters (demo.py)
DEMO_CHUNK_SIZE = 300
DEMO_CHUNK_OVERLAP = 30
DEMO_TOP_K = 3

# Evaluation parameters (run_evaluation.py)
CHUNK_SIZES = [300]
CHUNK_OVERLAPS = [30]
TOP_K_VALUES = [3]

# Prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""