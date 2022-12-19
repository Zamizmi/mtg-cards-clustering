# -*- coding: utf-8 -*-
import pandas as pd
from nltk.stem import PorterStemmer
from stop_words import MTG_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

porter_stemmer = PorterStemmer()


def stem_sentences(sentence: str):
    tokens = sentence.split()
    stop_word_free = [word for word in tokens if not word in MTG_STOP_WORDS]
    stemmed_tokens = [porter_stemmer.stem(token) for token in stop_word_free]
    return " ".join(stemmed_tokens)


def get_features_and_prepare_data(
    data: pd.DataFrame,
    remove_card_names=True,
    stemming=True,
    stop_words=True,
):
    prepared_data = data.copy()

    # many cards have no oracle text. Replace NaN with empty string
    prepared_data["oracle_text"] = prepared_data["oracle_text"].fillna("")

    # Replace card names with CARD_NAME
    if remove_card_names:
        prepared_data["oracle_text"] = prepared_data.apply(
            lambda x: str(x["oracle_text"]).replace(x["name"], ""),
            axis=1,
        )

    # handle stopwords
    if stop_words:
        prepared_data["oracle_text"] = prepared_data.apply(
            lambda x: str(x["oracle_text"]).replace(".", ""),
            axis=1,
        )

        prepared_data["oracle_text"] = prepared_data.apply(
            lambda x: str(x["oracle_text"]).replace(",", ""),
            axis=1,
        )

        prepared_data["oracle_text"] = prepared_data.apply(
            lambda x: " ".join(
                [
                    word
                    for word in x["oracle_text"].split()
                    if word.lower() not in (MTG_STOP_WORDS)
                ]
            ),
            axis=1,
        )

    # Normalise words
    if stemming:
        prepared_data["oracle_text"] = prepared_data["oracle_text"].apply(
            stem_sentences,
        )

    strip_accents = "unicode" if stop_words else None

    vectorizer = TfidfVectorizer(
        stop_words=None, strip_accents=strip_accents, lowercase=True
    )

    bag_of_words_vectorizer = CountVectorizer(
        stop_words=None, strip_accents=strip_accents, lowercase=True
    )

    vectorizer.fit(prepared_data["oracle_text"])
    text = vectorizer.transform(prepared_data["oracle_text"])
    features = vectorizer.fit_transform(prepared_data["oracle_text"])

    # if features_only:
    return features, text, vectorizer
    # featuresBoW = bag_of_words_vectorizer.fit_transform(prepared_data["oracle_text"])

    # distortions = []
    # distortionsBoW = []
    # K = range(kmin, kmax + 1)
    # for k in K:
    #     print(f"----- Here we go: {k}")
    #     kmeanModel = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init)
    #     kmeanModel.fit(features)
    #     distortions.append(kmeanModel.inertia_)

    #     kmeanModelBoW = KMeans(
    #         n_clusters=k, init=init, max_iter=max_iter, n_init=n_init
    #     )
    #     kmeanModelBoW.fit(featuresBoW)
    #     distortionsBoW.append(kmeanModelBoW.inertia_)

    # return [K, distortions, features, distortionsBoW, featuresBoW]
