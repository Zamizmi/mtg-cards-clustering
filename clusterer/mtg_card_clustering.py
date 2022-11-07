# -*- coding: utf-8 -*-


# Step 1: Load the data
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from stop_words import MTG_STOP_WORDS


def stem_sentences(sentence: str):
    tokens = sentence.split()
    stop_word_free = [word for word in tokens if not word in MTG_STOP_WORDS]
    stemmed_tokens = [porter_stemmer.stem(token) for token in stop_word_free]
    return " ".join(stemmed_tokens)


df = pd.read_json("./data/long_data.json", "values")


legal_df = df[df["legalities"].astype(str).str.contains("""'commander': 'legal'""")]
non_pw_legal_df = legal_df[~legal_df["type_line"].str.contains("Planeswalker")]
non_pw_legal_df.set_index("id", inplace=True)

# Drop rows with missing labels
non_pw_legal_df.dropna(subset=["oracle_text"], inplace=True)

# Replace card names with CARD_NAME to reduce the size of vocabulary
non_pw_legal_df["oracle_text"] = non_pw_legal_df.apply(
    lambda x: str(x["oracle_text"]).replace(x["name"], "CARD_NAME"),
    axis=1,
)

porter_stemmer = PorterStemmer()

non_pw_legal_df["oracle_text"] = non_pw_legal_df["oracle_text"].apply(
    stem_sentences,
)


# Step 2: Explore the data

# Step 3: Data preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Only select the Product and Consumer complaint columns
col = ["oracle_text", "name"]
data = non_pw_legal_df[col]

vectorizer = TfidfVectorizer(
    stop_words=MTG_STOP_WORDS, strip_accents="unicode", lowercase=True
)

features = vectorizer.fit_transform(data["oracle_text"])

k = 24
kmin = 1
kmax = 50
init = "k-means++"
max_iter = 400
n_init = 20

model = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init)
model.fit(features)

non_pw_legal_df["cluster"] = model.labels_

# output the result to a text file.

clusters = non_pw_legal_df.groupby("cluster")

for cluster in clusters.groups:
    f = open("results/cluster" + str(cluster) + ".csv", "w")  # create csv file
    data = clusters.get_group(cluster)[
        ["oracle_text", "name"]
    ]  # get title and overview columns
    f.write(data.to_csv(index_label="id", sep="€"))  # set index to id
    f.close()

print("Cluster centroids: \n")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# for i in range(k):
#     print("Cluster %d:" % i)
#     for j in order_centroids[i, :10]:  # print out 10 feature terms of each cluster
#         print(" %s" % terms[j])
#     print("------------")


def calculate_silhoutte_score(points, kmin: int, kmax: int):
    print("start")
    sil = []

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(kmin, kmax):
        print(f"----- Here we go: {k}")
        kmeans = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init).fit(
            points
        )
        labels = kmeans.labels_
        sil.append(silhouette_score(points, labels, metric="euclidean"))

    return sil


def calculate_WSS(points, kmin: int, kmax: int):
    sse = []
    for k in range(kmin, kmax):
        print(f"----- Here we go: {k}")
        kmeans = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init).fit(
            points
        )
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(points.shape[0]):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (
                points[i, 1] - curr_center[1]
            ) ** 2

        sse.append(curr_sse)
    return sse


# # silhouette_scores = calculate_silhoutte_score(features, kmin, kmax)


# wss_scores = calculate_WSS(features, kmin, kmax)

# plt.plot(wss_scores, "o-", color="green", label="WSS")
# # plt.plot(silhouette_scores, "o-", color="blue", label="Silhouette")

# plt.title(f"WSS and Silhouette scores between k={kmin} and k={kmax}")
# plt.legend(loc="lower right")
# plt.show()


def elbow_method(Y_sklearn):
    """
    This is the function used to get optimal number of clusters in order to feed to the k-means clustering algorithm.
    """

    number_clusters = range(
        kmin, kmax
    )  # Range of possible clusters that can be generated
    kmeans = [
        KMeans(n_clusters=i, init=init, max_iter=max_iter, n_init=n_init).fit(Y_sklearn)
        for i in number_clusters
    ]  # Getting no. of clusters

    score = [
        kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))
    ]  # Getting score corresponding to each cluster.
    score = [i * -1 for i in score]  # Getting list of positive scores.
    return (number_clusters, score)


distortions = []
K = range(kmin, kmax)
for k in K:
    print(f"----- Here we go: {k}")
    kmeanModel = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init)
    kmeanModel.fit(features)
    distortions.append(kmeanModel.inertia_)

# inertia here is referring to total sum of squares (or inertia) I of a cluster

plt.figure(figsize=(16, 8))
plt.plot(K, distortions, "bx-")
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("The Elbow Method showing the optimal k")
plt.show()


# val1 = elbow_method(features)
# fig, ax = plt.subplots()
# ax.set_xlim(kmin - 1, kmax)
# ax.plot(val1[0], val1[1], color="green")

# plt.xlabel("Number of Clusters")
# plt.ylabel("Score")
# plt.title("Elbow Method")
# plt.show()


# elbow_method(features)
