# from langchain import ChatGoogleGenerativeAI, PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core import prompts

from API_KEY_PRIVATE import GOOGLE_API_KEY


def init_llm(temperature, max_tokens, top_p, max_retries):
    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model="gemini-2.0-flash",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
    )
    llm_prompt_template = """You are Homer from the simpsons here to answer questions about the Simpsons Serie.
    The context provided has details and transcripts of a hundred of the simpsons episodes. Use it to answer the question   .
    Always cite the source metadata of the retrieved context.
    Act as if you were the character Homer from the scripts.
    Question: {question} \nContext: {context} \nAnswer:"""

    llm_prompt = prompts.PromptTemplate.from_template(llm_prompt_template)

    return llm, llm_prompt
