import nltk

# required_resources = [
#     'punkt', 
#     'punkt_tab', 
#     'averaged_perceptron_tagger',
#     'averaged_perceptron_tagger_eng',
#     'stopwords',
#     'wordnet',
#     'omw-1.4'
# ]

# for res in required_resources:
#     nltk.download(res)

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

from openai import OpenAI

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


    # client = OpenAI(api_key="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")
    # response = client.chat.completions.create(
    #     model="gpt-5-nano",
    #     input="write a haiku about ai",
    #     store=True,
    # )

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(model="text-embedding-3-small"), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
