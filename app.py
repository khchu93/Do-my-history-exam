"""
Streamlit web interface for the RAG Q&A system.
Mobile-responsive design optimized for Android phones.

Usage:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import time

from rag_system import RAGSystem
from config import PDF_PATH, DEMO_TOP_K


# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Board Game Q&A Assistant",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",  # Better for mobile
    menu_items={
        'Get Help': 'https://github.com/yourusername/rag-board-game-qa',
        'Report a bug': 'https://github.com/yourusername/rag-board-game-qa/issues',
        'About': "# RAG Q&A System\nPowered by GPT-3.5 and ChromaDB"
    }
)


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
    
    /* Question cards */
    .question-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .question-card:hover {
        background-color: #e0e2e6;
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
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


@st.cache_resource
def load_rag_system():
    """Load RAG system (cached to avoid reloading on every interaction)"""
    try:
        with st.spinner("🔄 Initializing AI system... This may take a minute."):
            rag = RAGSystem(
                pdf_path=str(PDF_PATH),
                chunk_size=300,
                chunk_overlap=30
            )
        return rag, None
    except Exception as e:
        return None, str(e)


def format_answer_with_sources(answer, context):
    """Format answer and sources in a mobile-friendly way"""
    # Display answer
    st.markdown(f"""
    <div class="answer-box">
        <h3>💡 Answer</h3>
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


def main():
    # Header
    st.title("🎲 Board Game Q&A Assistant")
    st.markdown("*Ask me anything about CATAN rules!*")
    
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
        time.sleep(0.5)  # Brief pause to show success message
        st.rerun()
    
    # Quick question suggestions (mobile-friendly cards)
    st.markdown("### 🔍 Try these questions:")
    
    example_questions = [
        "How do you win the game?",
        "What happens when you roll a 7?",
        "How do you build a settlement?",
        "What is the longest road?",
        "How many resource cards can you have?"
    ]
    
    # Create a grid of question buttons (2 per row on mobile)
    cols = st.columns(2)
    for idx, question in enumerate(example_questions):
        with cols[idx % 2]:
            if st.button(question, key=f"example_{idx}", use_container_width=True):
                st.session_state.current_question = question
    
    st.markdown("---")
    
    # Main input area
    st.markdown("### 💬 Ask your question:")
    
    # Text input
    user_question = st.text_input(
        "Type your question here...",
        value=st.session_state.get('current_question', ''),
        placeholder="e.g., How do I trade with other players?",
        label_visibility="collapsed"
    )
    
    # Search button
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process question
    if search_clicked and user_question:
        # Clear previous question from state
        if 'current_question' in st.session_state:
            del st.session_state.current_question
        
        # Add to chat history
        st.session_state.chat_history.insert(0, {
            'question': user_question,
            'timestamp': time.strftime("%H:%M")
        })
        
        # Get answer
        with st.spinner("🤔 Thinking..."):
            try:
                answer, context = st.session_state.rag_system.answer_question(
                    user_question,
                    k=DEMO_TOP_K,
                    return_context=True
                )
                
                # Display result
                format_answer_with_sources(answer, context)
                
                # Store in history
                st.session_state.chat_history[0]['answer'] = answer
                st.session_state.chat_history[0]['context'] = context
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Show chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Questions")
        
        # Limit to last 5 questions on mobile
        for i, chat in enumerate(st.session_state.chat_history[:5]):
            if i > 0:  # Skip the current question (already displayed above)
                with st.expander(f"🕐 {chat['timestamp']} - {chat['question']}", expanded=False):
                    if 'answer' in chat:
                        st.markdown(f"**Answer:** {chat['answer']}")
                        if st.checkbox(f"Show sources", key=f"sources_{i}"):
                            for j, source in enumerate(chat.get('context', []), 1):
                                st.markdown(f"**Source {j}:** {source[:200]}...")
    
    # Sidebar for settings (collapsible on mobile)
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Top-K selector
        k_value = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=DEMO_TOP_K,
            help="Higher values provide more context but may include less relevant information"
        )
        
        if k_value != DEMO_TOP_K:
            st.info(f"Using top-{k_value} retrieval")
        
        st.markdown("---")
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI assistant answers questions about CATAN using:
        - 🤖 GPT-3.5 for natural language understanding
        - 🔍 Semantic search for accurate information retrieval
        - 📚 Official CATAN rulebook as knowledge base
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit")


if __name__ == "__main__":
    main()