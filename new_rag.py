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


llm_prompt_template_2 = """You are Homer from the simpsons here to answer questions about the Simpsons Serie.
    You should rephrase the answer as if it was said by you, a character of the show.
    Always cite the source metadata of the retrieved context.
    Act as if you were the character Homer from the scripts.
    Question: {question} \nAnswer: {answer}"""


def query_rag(query_text):

    embedding_function = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
    )

    # récuper le vectorstore
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = db.as_retriever()

    prompt_template = prompts.PromptTemplate.from_template(llm_prompt_template_2)

    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")

    # MULTIPLE QUERIES
    # # retrieval_chain = init_multi_query_model(retriever)
    # final_rag_chain = (
    #     {"context": retrieval_chain, "question": itemgetter("question")}
    #     | prompt_template
    #     | model
    #     | StrOutputParser()
    # )

    final_rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | prompt_template
        | model
        | StrOutputParser()
    )
        # Execute the query and get the results
    result = final_rag_chain.invoke({"question": query_text})

    sources = [doc.metadata.get("source", None) for doc, _score in results]

    # Collect paragraphs from retrieved documents
    paragraphs = [doc.page_content for doc, _score in results]

    return {
        "answer": result,
        "paragraphs": paragraphs,
        "sources" : sources
    }


def query_rag(query_text):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    Args:
      - query_text (str): The text to query the RAG system with.
    Returns:
      - formatted_response (str): Formatted response including the generated text and sources.
      - response_text (str): The generated response text.
    """
   
    result = run(query_text)
    return result["answer"]

    # # Generate response text based on the prompt
    # response_text = model.invoke(prompt)

    # # Get sources of the matching documents
    # sources = [doc.metadata.get("source", None) for doc, _score in results]

    # # Format and return response including generated text and sources
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # return formatted_response, response_text


# test
response_text = query_rag("Who is Apu?")
print(response_text)
