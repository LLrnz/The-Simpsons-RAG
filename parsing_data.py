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
