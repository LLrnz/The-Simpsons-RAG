import generate_embeddings
from parsing_data import parsing_docs
from retriever import init_retriever


def test_generate_embeddings():
    docs = parsing_docs("https://simpsonscrazy.net/scripts/two-cars")
    embeddings = generate_embeddings.generate_embeddings(docs)


def test_retriever():
    docs = parsing_docs("https://simpsonscrazy.net/scripts/two-cars")
    embeddings = generate_embeddings.generate_embeddings(docs)
    retriever = init_retriever(embeddings)
    assert len(retriever.get_relevant_documents("MMLU")) > 0
