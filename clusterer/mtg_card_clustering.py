# -*- coding: utf-8 -*-


# Step 1: Load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data_preparement import get_features_and_prepare_data

MTG_CARD_DATA_FILE = "long_data_2022-12-21.json"

full_data = pd.read_json(f"./data/{MTG_CARD_DATA_FILE}", "values")

legal_data = full_data[
    full_data["legalities"].astype(str).str.contains("""'commander': 'legal'""")
]
non_pw_legal_data = legal_data[~legal_data["type_line"].str.contains("Planeswalker")]
non_pw_legal_data.set_index("id", inplace=True)

from sklearn.cluster import KMeans

k = 26  # the number of clusters used in the study. Please play around with the number
init = "k-means++"
max_iter = 400
n_init = 20

full_features, text, vectorizer = get_features_and_prepare_data(
    non_pw_legal_data, True, True, True
)


def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 2)

    sse = []
    for k in iters:
        sse.append(
            KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init)
            .fit(data)
            .inertia_
        )
        print("Fit {} clusters".format(k))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker="o")
    ax.set_xlabel("Cluster Centers")
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel("SSE")
    ax.set_title("SSE by Cluster Center Plot")
    plt.show()


# Way to approximate the wanted k-value.
# find_optimal_clusters(full_features, 60)


def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = range(data.shape[0])

    pca = PCA(n_components=2).fit_transform(data[max_items, :].todense())
    tsne = TSNE().fit_transform(
        PCA(n_components=50).fit_transform(data[max_items, :].todense())
    )

    idx = range(pca.shape[0])
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset, marker=".", s=[3])
    ax[0].set_title("PCA Cluster Plot")

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset, marker=".", s=[3])
    ax[1].set_title("TSNE Cluster Plot")

    plt.show()


clusters = KMeans(
    n_clusters=k, init=init, max_iter=max_iter, n_init=n_init
).fit_predict(full_features)

# Visualise the clustering results
plot_tsne_pca(full_features, clusters)


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    for i, r in df.iterrows():
        print("\nCluster {}".format(i))
        print(",".join([labels[t] for t in np.argsort(r)[-n_terms:]]))


# uncomment after a few rounds to print the actual top terms from every cluster
# get_top_keywords(text, clusters, vectorizer.get_feature_names(), 10)
