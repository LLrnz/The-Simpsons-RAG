# from langchain import ChatGoogleGenerativeAI, PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core import prompts


def init_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.\n
    Question: {question} \nContext: {context} \nAnswer:"""

    llm_prompt = prompts.PromptTemplate.from_template(llm_prompt_template)

    return llm, llm_prompt
