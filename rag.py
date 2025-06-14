from build_rag_chain import build


prompt_template = """Tu es un assistant dont la tache est de répondre à des questions.
    Utilise le contexte suivant pour y répondre. Le context est composé de 12 transcripts des épisodes de la saison 1 des simpsons.
    Chaque épisode commence par le titre de l'épisode, et est composé de plusieurs séquences.
    Si tu n'as pas la réponses, dis simplement que tu ne sais pas.
    Donne toujours le tag source des métadata des données et le titre de l'épisode source, que tu as utilisé pour répondre à la question.\n
    Question: {question} \nContexte: {context} \nRéponse:"""

rag_chain = build(
    [
        "https://simpsonspark.com/scripts/vf-nol-mortel",
        "https://simpsonspark.com/scripts/vf-bart_le_genie",
        "https://simpsonspark.com/scripts/vf-un_atome_de_bon_sens",
        "https://simpsonspark.com/scripts/vf-simpsonotherapie",
        "https://simpsonspark.com/scripts/vf-terreur_a_la_recre",
        "https://simpsonspark.com/scripts/vf-st-lisa-blues",
        "https://simpsonspark.com/scripts/vf-bart_a_perdu_la_tete",
        "https://simpsonspark.com/scripts/vf-marge-perd-la-boule",
        "https://simpsonspark.com/scripts/vf-l-odysse-d-homer",
        "https://simpsonspark.com/scripts/vf-l_espion_qui_venait_de_chez_moi",
        "https://simpsonspark.com/scripts/vf-un_clown_a_l_ombre",
        "https://simpsonspark.com/scripts/vf-une_soiree_d_enfer",
    ],
    prompt_template,
)
