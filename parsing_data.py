from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader


def parsing_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_content = docs[0].page_content

    text_content_1 = text_content.split("search site", 1)[1]
    final_text = text_content_1.split("OTHER WEBSITES:", 1)[0]

    docs = [Document(page_content=final_text, metadata={"source": "local"})]
    return docs


def parsing_doc_simpson_park(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_content = docs[0].page_content
    print(text_content)
    text_content_1 = text_content.split("Programme TV complet", 1)[1]
    final_text = text_content_1.split("L'équipe du site", 1)[0]

    docs = [Document(page_content=final_text, metadata={"source": url})]
    return docs


def parsing_docs_simpson_park(urls):
    loader = WebBaseLoader(urls)
    docs = loader.load()
    final_docs = []

    for i, doc in enumerate(docs):
        text_content = doc.page_content
        text_content_1 = text_content.split("Programme TV complet", 1)[1]
        final_text = text_content_1.split("L'équipe du site", 1)[0]
        final_docs.append(
            Document(page_content=final_text, metadata={"source": urls[i]})
        )

    return final_docs
