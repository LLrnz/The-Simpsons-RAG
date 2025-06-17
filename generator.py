# from langchain import ChatGoogleGenerativeAI, PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core import prompts


def init_llm(temperature, max_tokens, top_p, max_retries):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
    )
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.\n
    Question: {question} \nContext: {context} \nAnswer:"""

    llm_prompt = prompts.PromptTemplate.from_template(llm_prompt_template)

    return llm, llm_prompt
