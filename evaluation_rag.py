from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness
import time

import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY




def evaluate_retriever_precision_and_recall(retrieved_sources, expected_sources, k=3):
    
    #print("Sources récupérées :", retrieved_sources[:k])
    #print("Sources attendues :", expected_sources)
    somme_precision_at_k = 0.0

    for rank, source in enumerate(retrieved_sources[:k], start=1):
        if source in expected_sources:
            somme_precision_at_k += 1 / rank

    # on normalise par le nombre maximal de précision pondérée
    max_precision = sum(1 / (i + 1) for i in range(k))
    context_precision_at_k = somme_precision_at_k / max_precision

    relevant_retrieved = sum(1 for source in retrieved_sources[:k] if source in expected_sources)

    total_relevant = len(expected_sources)
    context_recall_at_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0

    return context_precision_at_k, context_recall_at_k
       

def evaluate_generation_queries(question, generated_answer, retrieved_paragraphs, llm):
    
    sample = SingleTurnSample(
        user_input=question,
        response=generated_answer,
        retrieved_contexts=retrieved_paragraphs
    )
    evaluator = Faithfulness(llm=LangchainLLMWrapper(llm))
    score = evaluator.single_turn_score(sample)
    return score





from new_rag import run
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

#liste pour stocker les résultats (question, réponse, faithfulness)
results_summary = []

for element in test_queries:
    question = element["query"]
    expected_sources = element["expected_docs"]
    print(f"\n Question : {question}")
    try:
        result = run(question)

        print("\nRéponse générée :")
        print(result["answer"])

        print("\nSources retenues :")
        for md in result["sources"]:
            print(md)

        """print("\nParagraphes extraits :")
        for p in result["paragraphs"]:
            print(p[:200] + "…")  # On affiche les 200 premiers caractères"""
    
        precision, recall = evaluate_retriever_precision_and_recall(result["sources"], expected_sources, 3)
        faithfulness = evaluate_generation_queries(question, result["answer"], result["paragraphs"], llm_instance)
        print(f"\nScore de la precision : {precision}")
        print(f"\nScore du recall: {recall}")
        print(f"\nScore de faithfulness : {faithfulness}")

        results_summary.append({
            "question": question,
            "answer": result["answer"],
            "precision" : precision,
            "recall" : recall,
            "faithfulness" : faithfulness
        })

        # on doit attendre pour pas dépasser le quota Gemini(nb de questions par minutes)
        time.sleep(30)

    except Exception as e:
        print(f"\nErreur lors du traitement de la question : {str(e)}")


print("\n\n Analyse générale :")
precision_scores = []
recall_scores = []
faithfulness_scores = []

for element in results_summary:
    faithfulness_scores.append(element["faithfulness"])
    
print("\n Etude du retriever :")
print(f'\n Moyenne de la précision : {sum(precision_scores) / len(precision_scores)}')
print(f'\n Moyenne du recall : {sum(recall_scores) / len(recall_scores)}')
print("\n Etude de la génération :")
print(f'\n Moyenne de la faithfulness  : {sum(faithfulness_scores) / len(faithfulness_scores)}')

