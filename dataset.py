import kagglehub
import os
import pandas as pd

SAVE_PATH = "simpson_dataset.csv"

base_path = kagglehub.dataset_download("prashant111/the-simpsons-dataset")

paths = {
    "episodes": os.path.join(base_path, "simpsons_episodes.csv"),
    "characters": os.path.join(base_path, "simpsons_characters.csv"),
    "scripts": os.path.join(base_path, "simpsons_script_lines.csv"),
}

df_episodes = pd.read_csv(paths["episodes"])
df_characters = pd.read_csv(paths["characters"])
df_scripts = pd.read_csv(paths["scripts"])

if 'character_id' in df_scripts.columns:
    df_scripts['character_id'] = pd.to_numeric(df_scripts['character_id'], errors='coerce')

print("Merging scripts with episodes...")
df_merged_episodes = pd.merge(
    df_scripts,
    df_episodes,
    left_on="episode_id",
    right_on="id",
    how="left",
    suffixes=("_script", "_episode")
)

if 'id_episode' in df_merged_episodes.columns:
    df_merged_episodes = df_merged_episodes.drop(columns=['id_episode'])

df_final_merged = pd.merge(
    df_merged_episodes,
    df_characters,
    left_on="character_id",
    right_on="id",
    how="left"
)

if 'id' in df_final_merged.columns and 'id_script' in df_final_merged.columns:
    df_final_merged = df_final_merged.drop(columns=['id'])

print(f"Saving merged dataset to {SAVE_PATH}...")
df_final_merged.to_csv(SAVE_PATH, index=False)

print(df_final_merged.head())
