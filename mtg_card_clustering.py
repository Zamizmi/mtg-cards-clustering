# -*- coding: utf-8 -*-


# Step 1: Load the data
import pandas as pd
import numpy as np

df = pd.read_json("./data/long_data.json", "values")

legal_df = df[df["legalities"].astype(str).str.contains("""'commander': 'legal'""")]
non_pw_legal_df = legal_df[~legal_df["type_line"].str.contains("Planeswalker")]
non_pw_legal_df.set_index("id", inplace=True)

# Drop rows with missing labels
non_pw_legal_df.dropna(subset=["oracle_text"], inplace=True)

non_pw_legal_df.info()

# Step 2: Explore the data

# Step 3: Data preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Only select the Product and Consumer complaint columns
col = ["oracle_text"]
data = non_pw_legal_df[col]


vectorizer = TfidfVectorizer(stop_words="english")
# TODO: make sure what the params are
# vectorizer = TfidfVectorizer(sublinear_tf= True, min_df=10, norm='l2', ngram_range=(1, 2), stop_words='english')

features = vectorizer.fit_transform(data["oracle_text"])

k = 30
model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1)
model.fit(features)

non_pw_legal_df["cluster"] = model.labels_


# output the result to a text file.

clusters = non_pw_legal_df.groupby("cluster")

# for cluster in clusters.groups:
#     f = open('cluster'+str(cluster)+ '.csv', 'w') # create csv file
#     data = clusters.get_group(cluster)[['oracle_text']] # get title and overview columns
#     f.write(data.to_csv(index_label='id')) # set index to id
#     f.close()

print("Cluster centroids: \n")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(k):
    print("Cluster %d:" % i)
    for j in order_centroids[i, :10]:  # print out 10 feature terms of each cluster
        print(" %s" % terms[j])
    print("------------")
