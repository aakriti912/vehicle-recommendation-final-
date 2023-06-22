import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data-set.csv")

features = ['brand']
for feature in features:
    df[feature] = df[feature].fillna('')


def combined_features(row):
    return row['categoryName'] + " " + row['categoryName'] + " " + row['subCategory'] + " " + row['subCategory'] + " " + row['brand'] + " " + row['name']


df["combined_features"] = df.apply(combined_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

product_id = 5865
product_index = df[df["id"] == product_id].index[0]

similar_products = list(enumerate(cosine_sim[product_index]))
sorted_similar_products = sorted(
    similar_products, key=lambda x: x[1], reverse=True)
sorted_similar_products

print(sorted_similar_products)
