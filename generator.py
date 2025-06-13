# from langchain import ChatGoogleGenerativeAI, PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core import prompts


def init_llm(prompt_template):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    llm_prompt = prompts.PromptTemplate.from_template(prompt_template)

    return llm, llm_prompt
