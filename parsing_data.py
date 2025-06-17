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

    # for i, doc in enumerate(docs):
    #     text_content = doc.page_content
    #     try:
    #         text_content_1 = text_content.split("Programme TV complet", 1)[1]
    #         final_text = text_content_1.split("L'équipe du site", 1)[0]
    #     except IndexError:
    #         print(f"Warning: Could not parse content from {urls[i]} as expected. Storing raw content.")
    #         final_text = text_content # Fallback to raw content if split fails

    #     final_docs.append(
    #         Document(page_content=final_text, metadata={"source": urls[i]})
    #     )

    return docs
