from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
import os


def parsing_docs(url):
    user_agent = os.getenv("USER_AGENT")
    headers = {"User-Agent": user_agent} if user_agent else {}
    loader = WebBaseLoader(url, requests_kwargs={"headers": headers})
    docs = loader.load()
    if not docs:
        raise ValueError(f"No content loaded from URL: {url}")
    text_content = docs[0].page_content

    text_content_1 = text_content.split("search site", 1)[1]
    final_text = text_content_1.split("OTHER WEBSITES:", 1)[0]

    docs = [Document(page_content=final_text, metadata={"source": "local"})]
    return docs


def parsing_doc_simpson_park(url):
    user_agent = os.getenv("USER_AGENT")
    headers = {"User-Agent": user_agent} if user_agent else {}
    loader = WebBaseLoader(url, requests_kwargs={"headers": headers})
    docs = loader.load()
    if not docs:
        raise ValueError(f"No content loaded from URL: {url}")
    text_content = docs[0].page_content
    text_content_1 = text_content.split("Programme TV complet", 1)[1]
    final_text = text_content_1.split("L'équipe du site", 1)[0]

    docs = [Document(page_content=final_text, metadata={"source": url})]
    return docs


def parsing_docs_simpson_park(urls):
    user_agent = os.getenv("USER_AGENT")
    headers = {"User-Agent": user_agent} if user_agent else {}
    loader = WebBaseLoader(urls, requests_kwargs={"headers": headers})
    docs = loader.load()
    final_docs = []

    for i, doc in enumerate(docs):
        text_content = doc.page_content
        try:
            text_content_1 = text_content.split("Programme TV complet", 1)[1]
            final_text = text_content_1.split("L'équipe du site", 1)[0]
        except IndexError:
            print(
                f"Warning: Could not parse content from {urls[i]} as expected. Storing raw content."
            )
            final_text = text_content  # Fallback to raw content if split fails

        final_docs.append(
            Document(page_content=final_text, metadata={"source": urls[i]})
        )

    return final_docs


def parsing_docs_simpson_archive(urls):
    user_agent = os.getenv("USER_AGENT")
    headers = {"User-Agent": user_agent} if user_agent else {}
    loader = WebBaseLoader(urls, requests_kwargs={"headers": headers})
    docs = loader.load()
    final_docs = []

    for i, doc in enumerate(docs):
        text_content = doc.page_content

        metadatas = doc.metadata

        writer = text_content.split("Written by ", 1)
        if len(writer) > 1:
            writer = writer[1]
            writer = writer.split("\n", 1)[0]
        else:
            writer = "not found"
        metadatas["writer"] = writer

        director = text_content.split("Directed by ", 1)
        if len(director) > 1:
            director = director[1]
            director = director.split("\n", 1)[0]
        else:
            director = "not found"
        metadatas["director"] = director

        airdate = text_content.split("Original airdate in N.A.: ", 1)
        if len(airdate) > 1:
            airdate = airdate[1]
            airdate = airdate.split("\n", 1)[0]
        else:
            airdate = "not found"
        metadatas["airdate"] = airdate

        voices = text_content.split("Voice credits", 1)
        if len(voices) > 1:
            voices = voices[1]
            voices = voices.split("Movie (and other) references", 1)[0]
        else:
            voices = "not found"

        metadatas["voices"] = transform_text_to_list(voices)

        final_docs.append(Document(page_content=text_content, metadata=metadatas))

    return final_docs


def transform_text_to_list(text):
    lines = text.strip().split("\n")
    names = []
    for line in lines:
        line = line.strip()
        if (
            line.startswith("-")
            and not line.startswith("- Starring")
            and not line.startswith("- Special Guest Voices")
            and not line.startswith("- Also Starring")
        ):
            name = line[1:].strip()
            names.append(name)
    return str(names)
