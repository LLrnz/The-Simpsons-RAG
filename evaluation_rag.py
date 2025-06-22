from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness
import time

import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



def evaluate_retriever_queries(question, generated_answer, retrieved_paragraphs, llm):
    sample = SingleTurnSample(
        user_input=question,
        response=generated_answer,
        retrieved_contexts=retrieved_paragraphs
    )
    evaluator_precision =LLMContextPrecisionWithReference(llm=LangchainLLMWrapper(llm))
    evaluator_recall = LLMContextRecall(llm=LangchainLLMWrapper(llm))
   
    precision_score = evaluator_precision.single_turn_ascore(sample)
    recall_score = evaluator_recall.single_turn_ascore(sample)
    
    return precision_score, recall_score



def evaluate_generation_queries(question, generated_answer, retrieved_paragraphs, llm):
    
    sample = SingleTurnSample(
        user_input=question,
        response=generated_answer,
        retrieved_contexts=retrieved_paragraphs
    )
    evaluator = Faithfulness(llm=LangchainLLMWrapper(llm))
    score = evaluator.single_turn_score(sample)
    return score




from rag import rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI
llm_instance = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


test_queries = [
    {
        "query": 'In which episode does Marge say: "Have you been up all night eating cheese?"',
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/1F01.html"],
    },
    {
        "query": "Which characters are present in the episode $pringfield?",
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/1F08.html"],
    },
    {
        "query": "Who is the director of  Lisa's Date With Density?",
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/4F01.html"],
    },
    {
        "query": "When was Maggie born?",
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/2F10.html"],
    },
    {
        "query": "What does Hommer do so he can work from home?",
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/3F05.html"],
    },
    {
        "query": "Whith whom Bart goes to tennessee?",
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/3F17.html"],
    },
    {
        "query": "What are the funfacts about Sideshow Bob Roberts?",
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/2F02.html"],
    },
    {
        "query": "In which episodes does Wiggum appears?",
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/2F21.html"],
    },
    {
        "query": "Who is the main suspect of Mr. Burns murder? I do not want the true culprit.",
        "expected_docs": [
            "https://www.simpsonsarchive.com/episodes/2F20.html",
            "https://www.simpsonsarchive.com/episodes/2F16.html",
        ],
    },
    {
        "query": "Who dubs Burns?",
        "expected_docs": ["https://www.simpsonsarchive.com/episodes/1F16.html"],
    },
]

#liste pour stocker les résultats (question, réponse, precision, recall, faithfulness)
results_summary = []

for element in test_queries:
    question = element["query"]
    print(f"\n Question : {question}")
    try:
        résultat = rag_chain({"query": question})

        answer = résultat["result"]
        sources = résultat.get("source_documents", [])
        paragraphs = [doc.page_content for doc in sources]
        

        print("\nRéponse générée :")
        print(answer)

        print("\nSources retenues :")
        for md in sources:
            print(md)

    
        context_precision, context_recall = evaluate_retriever_queries(question, answer, paragraphs, llm_instance)
        faithfulness = evaluate_generation_queries(question, answer, paragraphs, llm_instance)
        print(f"\nScore de precision : {context_precision}")
        print(f"\nScore de recall : {context_recall}")
        print(f"\nScore de faithfulness : {faithfulness}")

        results_summary.append({
            "question": question,
            "answer": answer,
            "precision": context_precision,
            "recall" : context_recall,
            "faithfulness" : faithfulness
        })

        # on doit attendre pour pas dépasser le quota Gemini(nb de questions par minutes)
        time.sleep(30)

    except Exception as e:
        print(f"\nErreur lors du traitement de la question : {str(e)}")


print("\n\n Analyse générale :")
precisions_scores = []
recall_scores =[]
faithfulness_scores = []

for element in results_summary:
    precisions_scores.append(element["precision"])
    recall_scores.append(element["recall"])
    faithfulness_scores.append(element["faithfulness"])
    
print("\n Etude du retriever :")
print(f'\n précision moyenne : {sum(precisions_scores)/len(precisions_scores)} ')
print(f'\n recall moyen : {sum(recall_scores)/len(recall_scores)}')
print("\n Etude de la génération :")
print(f'\n faithfulness moyenne : {sum(faithfulness_scores) / len(faithfulness_scores)}')

