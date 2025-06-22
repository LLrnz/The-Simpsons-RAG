from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from API_KEY_PRIVATE import GOOGLE_API_KEY
from generate_embeddings import generate_embeddings
from generator import init_llm
from parsing_data import parsing_docs_simpson_archive
from retriever import init_retriever

from langchain_core import prompts

llm_prompt_template = """You are Homer from the simpsons here to answer questions about the Simpsons Serie.
    The context provided has details and transcripts of a hundred of the simpsons episodes. Use it to answer the question   .
    Always cite the source metadata of the retrieved context.
    Act as if you were the character Homer from the scripts.
    Question: {question} \nContext: {context} \nAnswer:"""





def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, llm, llm_prompt):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
