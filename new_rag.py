from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import StrOutputParser
from operator import itemgetter

from API_KEY_PRIVATE import GOOGLE_API_KEY


from langchain_core import prompts

from decomposition import invoke_decomposition
from multi_query import init_multi_query_model

llm_prompt_template = """You are Homer from the simpsons here to answer questions about the Simpsons Serie.
    The context provided has details and transcripts of a hundred of the simpsons episodes. Use the context to answer the question.
    Always cite the source metadata of the retrieved context.
    Act as if you were the character Homer from the scripts.
    Question: {question} \nContext: {context} \nAnswer:"""


def query_rag(query_text):

    embedding_function = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
    )

    # récuper le vectorstore
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = db.as_retriever()

    prompt_template = prompts.PromptTemplate.from_template(llm_prompt_template)

    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")

    # MULTIPLE QUERIES
    # retrieval_chain = init_multi_query_model(retriever)

    # DECOMPOSED QUERY
    retrieval_chain = invoke_decomposition(query_text, retriever)

    final_rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | prompt_template
        | model
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"question": query_text})


# test
response_text = query_rag("Who is Apu?")
print(response_text)
