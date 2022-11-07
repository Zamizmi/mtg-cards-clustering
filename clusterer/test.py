import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_json("./data/long_data.json", "values")


legal_df = df[df["legalities"].astype(str).str.contains("""'commander': 'legal'""")]
non_pw_legal_df = legal_df[~legal_df["type_line"].str.contains("Planeswalker")]

non_pw_legal_df["oracle_text"] = non_pw_legal_df.apply(
    lambda x: str(x["oracle_text"]).replace("Static Orb", "CARD_NAME"),
    axis=1,
)


non_pw_legal_df.set_index("id", inplace=True)

print(non_pw_legal_df["oracle_text"].head())
