"""
Interactive demo script for the RAG system.

Usage:
    python demo.py
"""

import sys
from pathlib import Path

from src.rag_system import RAGSystem
from src.config import PDF_PATH, DEMO_TOP_K, PROMPT_TEMPLATE, DEMO_CHUNK_SIZE, DEMO_CHUNK_OVERLAP


def print_header():
    """Print demo header."""
    print("\n" + "=" * 70)
    print("   Board Game Manual Q&A System")
    print("   Powered by RAG (Retrieval-Augmented Generation)")
    print("=" * 70 + "\n")


def print_example_questions():
    """Print example questions."""
    print("Example questions you can ask:")
    print("  • How do you win the game?")
    print("  • What happens when you roll a 7?")
    print("  • How many resource cards can you have?")
    print("  • What is the longest road?")
    print("  • How do you build a settlement?")
    print("\n")


def main():
    """Run interactive demo."""
    print_header()
    
    # Check if PDF exists
    if not Path(PDF_PATH).exists():
        print(f"❌ Error: PDF not found at {PDF_PATH}")
        print("Please ensure the PDF file is in the correct location.")
        sys.exit(1)
    
    # Initialize RAG system
    print("🔄 Initializing RAG system...")
    print(f"   Loading: {PDF_PATH}")
    
    try:
        rag = RAGSystem(
            pdf_path=str(PDF_PATH), 
            chunk_size=DEMO_CHUNK_SIZE, 
            chunk_overlap=DEMO_CHUNK_OVERLAP)
        print("✅ System ready!\n")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        sys.exit(1)
    
    print_example_questions()
    
    # Interactive loop
    print("Type 'quit' or 'exit' to stop, 'help' for example questions.\n")
    
    while True:
        try:
            # Get user input
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print()
                print_example_questions()
                continue
            
            # Answer the question
            print("\n🔍 Searching for relevant information...")
            answer, context = rag.answer_question(
                question, 
                k=DEMO_TOP_K,
                return_context=True,
                prompt=PROMPT_TEMPLATE
            )
            
            # Display answer
            print("\n" + "─" * 70)
            print("💡 ANSWER:")
            print("─" * 70)
            print(answer)
            print("─" * 70)
            
            # Optionally show sources
            show_sources = input("\n📚 Show source passages? (y/n): ").strip().lower()
            if show_sources == 'y':
                print("\n" + "─" * 70)
                print("📖 SOURCE PASSAGES:")
                print("─" * 70)
                for i, ctx in enumerate(context, 1):
                    print(f"\n[{i}] {ctx[:200]}..." if len(ctx) > 200 else f"\n[{i}] {ctx}")
                print("─" * 70)
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing question: {e}\n")
    
    # Cleanup
    rag.cleanup()


if __name__ == "__main__":
    main()