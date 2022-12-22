# Bachelors thesis: Clustering MTG-cards

This is part of my bachelors thesis.

You can download and try the clustering yourself by downloading the bulk data from [Scryfall](https://scryfall.com/docs/api/bulk-data)

## Steps to reproduce

### Environment

Please set up your virtual environment for Python by following these [instructions](https://docs.python.org/3/tutorial/venv.html)

After setting up the environment, activate it with
`source bin/activate`

And then install the needed packages with pip
`pip install -r requirements.txt`

### Clustering the cards

Steps

1. Move the cards in json-format to data/folder, and name as you wish eg. `mtg_card_data.json``
2. Rename MTG_CARD_DATA_FILE variable in the file `/clusterer/mtg_card_clustering.py`
3. run `python clusterer/mtg_card_clustering.py`
4. View the elbow-method graph by uncommenting the line before `find_optimal_clusters(full_features, 60)` and change the value 60 as you please. Do note the higher the value, the longer it takes to process.
5. To view results of every cluster, uncomment the line before `get_top_keywords(text, clusters, vectorizer.get_feature_names(), 10)`

If you find the work interesting or have feedback regarding the work, please send a message in [linkedin](https://www.linkedin.com/in/sami-lindqvist/)
