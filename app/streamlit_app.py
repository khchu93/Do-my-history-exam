"""
Streamlit web interface for the RAG Q&A system.
Mobile-responsive design optimized for Android phones.

Usage:
    streamlit run app.py
"""


import streamlit as st
from pathlib import Path
import time
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from src.config import PDF_PATH, DEMO_CHUNK_SIZE, DEMO_CHUNK_OVERLAP, DEMO_TOP_K, PROMPT_TEMPLATE, SIMILARITY_SEARCH

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Board Game Q&A Assistant",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/yourusername/rag-board-game-qa',
        'Report a bug': 'https://github.com/yourusername/rag-board-game-qa/issues',
        'About': "# RAG Q&A System\nPowered by GPT-4 and ChromaDB"
    }
)

# Constants
DEFAULT_QUESTION = "How do I trade with other players?"
RULEBOOK_URL = "https://www.catan.com/sites/default/files/2025-03/CN3081%20CATAN%E2%80%93The%20Game%20Rulebook%20secure%20%281%29.pdf"

# Custom CSS for mobile responsiveness
st.markdown("""
<style>            
    /* Mobile-first responsive design */
    .main {
        padding: 1rem;
    }
    
    /* Better text readability on mobile */
    .stMarkdown {
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Larger buttons for touch screens */
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 18px;
        margin: 0.5rem 0;
    }
    
    /* Better input fields on mobile */
    .stTextInput>div>div>input {
        font-size: 16px;
        padding: 0.75rem;
    }
    
    /* Expandable sections for better mobile UX */
    .streamlit-expanderHeader {
        font-size: 18px;
        font-weight: 600;
    }
    
    /* Answer styling */
    .answer-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Source passages */
    .source-box {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 14px;
    }
    
    /* Hide Streamlit branding on mobile for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Loading spinner styling */
    .stSpinner > div {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'last_qa' not in st.session_state:
        st.session_state.last_qa = None
    if 'last_processed_question' not in st.session_state:
        st.session_state.last_processed_question = ''
    if 'processing' not in st.session_state:
        st.session_state.processing = False


@st.cache_resource
def load_rag_system():
    """Load RAG system (cached to avoid reloading on every interaction)"""
    try:
        with st.spinner("🔄 Initializing AI system... This may take a minute."):
            rag = RAGSystem(
                chunk_size=DEMO_CHUNK_SIZE, 
                chunk_overlap=DEMO_CHUNK_OVERLAP,
                similarity_search=SIMILARITY_SEARCH
            )
        return rag, None
    except Exception as e:
        return None, str(e)


def format_answer_with_sources(question, answer, context):
    """Format question, answer and sources in a mobile-friendly way"""
    # Display question and answer in the same container
    st.markdown(f"""
    <div class="answer-box">
        <h4>❓ Question</h4>
        <p>{question}</p>
        <hr style="margin: 1rem 0; border: none; border-top: 1px solid #ccc;">
        <h4>💡 Answer</h4>
        <p>{answer}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display sources in expandable section
    with st.expander("📚 View Source Passages", expanded=False):
        for i, source in enumerate(context, 1):
            # Truncate long sources on mobile
            display_text = source if len(source) < 300 else source[:300] + "..."
            st.markdown(f"""
            <div class="source-box">
                <strong>Source {i}:</strong><br>
                {display_text}
            </div>
            """, unsafe_allow_html=True)


def get_answer(question):
    """Get answer for a question and store in session state"""
    try:
        answer, context = st.session_state.rag_system.answer_question(
            question,
            k=DEMO_TOP_K,
            return_context=True,
            prompt=PROMPT_TEMPLATE
        )
        
        # Store the Q&A in session state
        st.session_state.last_qa = {
            'question': question,
            'answer': answer,
            'context': context
        }
        st.session_state.processing = False
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.session_state.last_qa = None
        st.session_state.processing = False


def handle_question(question):
    """Process a question if it's new"""
    question = question.strip()
    
    # Use default question if empty
    if not question:
        question = DEFAULT_QUESTION
    
    # Only process if it's a new question
    if question != st.session_state.last_processed_question:
        st.session_state.last_processed_question = question
        
        # Clear previous Q&A immediately and set processing flag
        st.session_state.last_qa = None
        st.session_state.processing = True
        
        get_answer(question)
        return True 
    return False


def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("## 🎲 Board Game Q&A Assistant (CATAN)")
    st.markdown("*Catan is a strategy game where you explore a new island, collect resources from each dice roll, and build roads, settlements, and cities while trading with other players.*")
    st.markdown("*The board changes every game, and each turn opens new choices, so it stays lively and unpredictable.*")
    st.markdown("*But the rules and edge cases can confuse newcomers, so I built a RAG game master that answers questions instantly and keeps the game flowing without anyone digging through the rulebook.*")
    st.markdown("\n*Ask me anything about CATAN rules!*")
    
    # Link button
    st.link_button(
    "📚 Open CATAN Rulebook (PDF)",
    "https://www.catan.com/sites/default/files/2025-03/CN3081%20CATAN%E2%80%93The%20Game%20Rulebook%20secure%20%281%29.pdf"
)

    # Initialize system
    if not st.session_state.initialized:
        rag, error = load_rag_system()
        if error:
            st.error(f"❌ Failed to initialize system: {error}")
            st.info("💡 Make sure your PDF and .env file are configured correctly.")
            st.stop()
        st.session_state.rag_system = rag
        st.session_state.initialized = True
        st.success("✅ System ready!")
        time.sleep(0.5)
        st.rerun()
    
    # Main input area
    st.markdown("#### 💬 What would you like to know about this game?")
    
    # Form for Enter key support and auto-clear
    with st.form(key="search_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_question = st.text_input(
                "question_input",
                placeholder=f"e.g., {DEFAULT_QUESTION}",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button(
                "🔍 Search",
                type="primary",
                use_container_width=True
            )
    
    # Handle form submission
    if submit_button:
        question_to_ask = user_question.strip() if user_question.strip() else DEFAULT_QUESTION
        if handle_question(question_to_ask):
            st.rerun()
    
    # Show spinner while processing
    if st.session_state.processing or (st.session_state.last_qa is None and st.session_state.last_processed_question):
        with st.spinner("🤔 Thinking..."):
            time.sleep(0.1)
    
    # Display last Q&A if it exists
    if st.session_state.last_qa:
        format_answer_with_sources(
            st.session_state.last_qa['question'],
            st.session_state.last_qa['answer'],
            st.session_state.last_qa['context']
        )
    
    # Quick question suggestions
    st.markdown("#### 🔍 Popular Questions:")
    
    example_questions = [
        "How do you win the game?",
        "What happens when you roll a 7?",
        "How do you build a settlement?",
        "What is the longest road?",
        "How many resource cards can you have?"
    ]
    
    # Create a grid of question buttons
    cols = st.columns(2)
    for idx, question in enumerate(example_questions):
        with cols[idx % 2]:
            if st.button(question, key=f"example_{idx}", use_container_width=True):
                if handle_question(question):
                    st.rerun()
    
    st.markdown("---")


if __name__ == "__main__":
    main()
