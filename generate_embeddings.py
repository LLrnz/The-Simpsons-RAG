from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from API_KEY_PRIVATE import GOOGLE_API_KEY
from langchain_community.vectorstores import Chroma


def generate_embeddings(docs):
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_API_KEY

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Save to disk
    vectorstore = Chroma.from_documents(
        documents=docs,  # Data
        embedding=gemini_embeddings,  # Embedding model
        persist_directory="./chroma_db",  # Directory to save data
    )

    return gemini_embeddings
