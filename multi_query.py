from langchain_core import prompts
from langchain.schema import StrOutputParser
from langchain.load import dumps, loads
from langchain_google_genai import ChatGoogleGenerativeAI

from API_KEY_PRIVATE import GOOGLE_API_KEY


def init_multi_query_model(vector_store):
    # Multi Query: Different Perspectives. Source : https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
    template = """You are an AI language model assistant. Your task is to generate five 
  different versions of the given user question to retrieve relevant documents from a vector 
  database. By generating multiple perspectives on the user question, your goal is to help
  the user overcome some of the limitations of the distance-based similarity search. 
  Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = prompts.PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY, model="gemini-2.0-flash", temperature=0
    )
    retriever = vector_store.as_retriever()
    generate_queries = (
        prompt_perspectives | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )
    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    return retrieval_chain


# Source  :
# Multi Query: Different Perspectives. Source : https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]
