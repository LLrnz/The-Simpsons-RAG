from langchain_core import prompts
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

from API_KEY_PRIVATE import GOOGLE_API_KEY


def decomposition_querries(question):
    # prompt source : https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
    response_prompt_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
    response_prompt = prompts.PromptTemplate.from_template(response_prompt_template)

    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY, model="gemini-2.0-flash", temperature=0
    )

    chain = response_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

    return chain.invoke({"question": question})


def answer_questions(questions, retriever):
    # Prompt source : https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = prompts.PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY, model="gemini-2.0-flash", temperature=0
    )

    q_a_pairs = ""
    for q in questions:

        rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs"),
            }
            | decomposition_prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
        q_a_pair = make_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
    return q_a_pairs


def make_qa_pair(question, answer):
    formatted_string = f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


def invoke_decomposition(query, retriever):
    questions = decomposition_querries(query)
    context = answer_questions(questions, retriever)

    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = prompts.PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY, model="gemini-2.0-flash", temperature=0
    )

    final_rag_chain = prompt | llm | StrOutputParser()

    return final_rag_chain
