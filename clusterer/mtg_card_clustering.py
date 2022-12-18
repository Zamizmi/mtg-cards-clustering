# -*- coding: utf-8 -*-


# Step 1: Load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preparement import get_features_and_prepare_data

full_data = pd.read_json("./data/long_data.json", "values")

legal_data = full_data[
    full_data["legalities"].astype(str).str.contains("""'commander': 'legal'""")
]
non_pw_legal_data = legal_data[~legal_data["type_line"].str.contains("Planeswalker")]
non_pw_legal_data.set_index("id", inplace=True)


# Drop rows with missing labels
# non_pw_legal_data.dropna(subset=["oracle_text"], inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

k = 24
kmin = 2
kmax = 50
init = "k-means++"
max_iter = 400
n_init = 20

# fully_prepared = get_features_and_prepare_data(
#     non_pw_legal_data, kmin, kmax, init, max_iter, n_init, True, False, True
# )

# features = fully_prepared[1]

# model = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init)
# model.fit(features)

# non_pw_legal_df["cluster"] = model.labels_

# output the result to a text file.

# clusters = non_pw_legal_df.groupby("cluster")

# for cluster in clusters.groups:
#     f = open("results/cluster" + str(cluster) + ".csv", "w")  # create csv file
#     data = clusters.get_group(cluster)[
#         ["oracle_text", "name"]
#     ]  # get title and overview columns
#     f.write(data.to_csv(index_label="id", sep="€"))  # set index to id
#     f.close()

# print("Cluster centroids: \n")
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names_out()

# for i in range(k):
#     print("Cluster %d:" % i)
#     for j in order_centroids[i, :10]:  # print out 10 feature terms of each cluster
#         print(" %s" % terms[j])
#     print("------------")

fully_prepared = get_features_and_prepare_data(
    non_pw_legal_data, kmin, kmax, init, max_iter, n_init, True, True, True
)

no_preparements = get_features_and_prepare_data(
    non_pw_legal_data, kmin, kmax, init, max_iter, n_init, False, False, False
)

plt.figure(figsize=(16, 8))
plt.plot(fully_prepared[0], fully_prepared[1], "bx-")
plt.plot(no_preparements[0], no_preparements[1], "rx-")
plt.plot(fully_prepared[0], fully_prepared[3], "gx-")
plt.plot(no_preparements[0], no_preparements[3], "yx-")
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("The Elbow Method showing the optimal k")
plt.show()
