import csv

from API_KEY_PRIVATE import GOOGLE_API_KEY
from generate_embeddings import generate_embeddings
from parsing_data import parsing_docs_simpson_archive
from retriever import init_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_retriever(url):
    docs = parsing_docs_simpson_archive(url)
    embeddings = generate_embeddings(docs)
    retriever = init_retriever(embeddings, len(url))

    return retriever


# GENERATE vector store
simpsons_archive = []

with open("scraped_scripts.csv", "r") as file:
    reader = csv.reader(file)
    for line in reader:
        simpsons_archive.append(line[1])
    simpsons_archive = simpsons_archive[1:30]

create_retriever(simpsons_archive)
