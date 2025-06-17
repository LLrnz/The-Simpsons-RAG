from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from API_KEY_PRIVATE import GOOGLE_API_KEY


def generate_embeddings(docs):
    # GOOGLE_API_KEY is expected to be in the environment,
    # loaded by dotenv in main.py or set by the deployment environment.
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
    )

    # Save to disk
    vectorstore = Chroma.from_documents(
        documents=docs,  # Data
        embedding=gemini_embeddings,  # Embedding model
        persist_directory="./chroma_db",  # Directory to save data
    )

    return gemini_embeddings
