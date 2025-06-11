from langchain_chroma import Chroma


def init_retriever(gemini_embeddings):
    vectorstore_disk = Chroma(
        persist_directory="./chroma_db",  # Directory of db
        embedding_function=gemini_embeddings,  # Embedding model
    )
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

    return retriever
