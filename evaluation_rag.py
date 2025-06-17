import csv
from db import catalogue
from build_rag_chain import create_retriever
from build_rag_chain import build


# evaluation du retriever


# precision
def calculate_context_precision_at_k(retrieved_docs, expected_docs, k=3):
    somme_precision_at_k = 0.0

    for rank, doc in enumerate(retrieved_docs[:k], start=1):
        # on vérifie si le document est pertinent
        if doc.metadata["source"] in expected_docs:
            somme_precision_at_k += 1 / rank

    # on normalise par le nombre maximal de précision pondérée
    max_precision = sum(1 / (i + 1) for i in range(k))
    context_precision_at_k = somme_precision_at_k / max_precision

    return context_precision_at_k


# recall
def calculate_context_recall_at_k(retrieved_docs, expected_docs, k=3):
    # Compter le nombre de documents pertinents récupérés dans les k premiers résultats
    relevant_retrieved = sum(
        1 for doc in retrieved_docs[:k] if doc.metadata["source"] in expected_docs
    )

    # Calculer le rappel du contexte
    total_relevant = len(expected_docs)
    context_recall_at_k = (
        relevant_retrieved / total_relevant if total_relevant > 0 else 0
    )

    return context_recall_at_k


def evaluate_retriever_precision_and_recall(retriever, test_queries, k=3):
    precisions = []
    recalls = []

    for test in test_queries:
        query = test["query"]
        expected_docs = test["expected_docs"]

        retrieved_docs = retriever.get_relevant_documents(query)

        precision_at_k = calculate_context_precision_at_k(
            retrieved_docs, expected_docs, k
        )
        precisions.append(precision_at_k)

        recall_at_k = calculate_context_recall_at_k(retrieved_docs, expected_docs, k)
        recalls.append(recall_at_k)

        print(f"Query: {query}")
        sources = []
        for doc in retrieved_docs[:k]:
            sources.append(doc.metadata["source"])
        print(f"expected : {expected_docs}, retrieved : {sources}")
        print(f"Context Precision@{k}: {precision_at_k:.2f}")
        print(f"Context Recall@{k}: {recall_at_k:.2f}")

    # précision moyenne et le rappel moyen
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)

    print(f"Average Context Precision@{k}: {average_precision:.2f}")
    print(f"Average Context Recall@{k}: {average_recall:.2f}")


# évaluation de la génération


def calculate_faithfulness_score(response, context):
    affirmations = response.split(". ")

    consistent_affirmations = sum(
        1 for affirmation in affirmations if affirmation.lower() in context.lower()
    )

    if affirmations:
        faithfulness_score = consistent_affirmations / len(affirmations)
    else:
        return 0

    return faithfulness_score


def evaluate_faithfulness(retriever, test_queries):
    rag_chain = build(retriever=retriever, temperature=0.1, top_p=0.1)
    faithfulness_scores = []

    for test in test_queries:
        query = test["query"]
        expected_docs = test["expected_docs"]

        retrieved_docs = retriever.get_relevant_documents(query)
        response = rag_chain.invoke(query)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        faithfulness_score = calculate_faithfulness_score(response, context)
        faithfulness_scores.append(faithfulness_score)

        print(f"Query: {query}")
        print(f"Faithfulness Score: {faithfulness_score:.2f}")

    average_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    print(f"Average Faithfulness Score: {average_faithfulness:.2f}")


simpsons_archive = []

with open("scraped_scripts.csv", "r") as file:
    reader = csv.reader(file)
    for line in reader:
        simpsons_archive.append(line[1])
    simpsons_archive = simpsons_archive[1:100]

retriever = create_retriever(simpsons_archive, 3)

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

print("évaluation du RAG : \n")
evaluate_retriever_precision_and_recall(retriever, test_queries, k=3)
print("évaluation de la génération: \n")
evaluate_faithfulness(retriever, test_queries)
