from parsing_data import parsing_doc_simpson_park, parsing_docs_simpson_park


def test_parsing_docs():
    print(parsing_doc_simpson_park("https://simpsonspark.com/scripts/vf-nol-mortel"))


def test_parsing_all_docs():
    print(
        parsing_docs_simpson_park(
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
            ]
        )
    )
