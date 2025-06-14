from langchain_community.vectorstores import Chroma


def init_retriever(gemini_embeddings, len_url):
    vectorstore_disk = Chroma(
        persist_directory="./chroma_db",  # Directory of db
        embedding_function=gemini_embeddings,  # Embedding model
    )
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": len_url})

    return retriever
