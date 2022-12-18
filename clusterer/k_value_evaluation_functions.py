from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from stop_words import MTG_STOP_WORDS


def calculate_silhoutte_score(
    points, KMeans, kmin: int, kmax: int, init: int, max_iter: int, n_init: int
):
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


def calculate_WSS(
    points, KMeans, kmin: int, kmax: int, init: int, max_iter: int, n_init: int
):
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


def elbow_method(
    features, KMeans, kmin: int, kmax: int, init: int, max_iter: int, n_init: int
):
    """
    This is the function used to get optimal number of clusters in order to feed to the k-means clustering algorithm.
    """

    number_clusters = range(
        kmin, kmax
    )  # Range of possible clusters that can be generated
    kmeans = [
        KMeans(n_clusters=i, init=init, max_iter=max_iter, n_init=n_init).fit(features)
        for i in number_clusters
    ]  # Getting no. of clusters

    score = [
        kmeans[i].fit(features).score(features) for i in range(len(kmeans))
    ]  # Getting score corresponding to each cluster.
    score = [i * -1 for i in score]  # Getting list of positive scores.
    return (number_clusters, score)
