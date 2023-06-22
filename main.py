from typing import Union
from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

df = pd.read_csv("data-set.csv")

features = ['brand']
for feature in features:
    df[feature] = df[feature].fillna('')


def combined_features(row):
    name = row["name"]
    categoryName = row["categoryName"].replace(" ", "_")
    subCategory = row["subCategory"]
    brand = row["brand"]

    return f"{ categoryName } { categoryName } { subCategory } { subCategory } { brand } { name }".lower()


df["combined_features"] = df.apply(combined_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)


@app.get("/recommendations/{product_id}")
def get_recommendations(product_id: int):
    product_index = df[df["id"] == product_id].index[0]

    similar_products = list(enumerate(cosine_sim[product_index]))
    sorted_similar_products = sorted(
        similar_products, key=lambda x: x[1], reverse=True)

    result = [{"productId": int(df.iloc[item[0]]["id"]), "similarity": item[1]}
              for item in sorted_similar_products if item[1] > 0 and int(df.iloc[item[0]]["id"]) != product_id]

    return result[0:10]


# @app.post("/products")
# def add_product(product_data: dict):
#     df.loc[len(df)] = product_data

#     # Calculate the feature representation for the new product
#     combined_features = combined_features(product_data)
#     new_product_feature_vector = cv.transform([combined_features])

#     # Compute similarity between the new product and existing products
#     similarity_scores = cosine_similarity(
#         count_matrix, new_product_feature_vector)

#     # Update the similarity matrix
#     global cosine_sim
#     cosine_sim = np.append(cosine_sim, similarity_scores, axis=1)

#     return {"message": "Product added successfully."}

def add_product(product_data: dict):
    # Assuming product_data contains the features of the new product
    # Add the new product to the dataset
    product_id = product_data["id"]

    # Check if the product with the same ID already exists
    if df["id"].isin([product_id]).any():
        return "product with this id already exists"

    df.loc[len(df)] = product_data

    # Calculate the feature representation for the new product
    product_combined_features = combined_features(product_data)
    new_product_feature_vector = cv.transform([product_combined_features])

    # Compute similarity between the new product and existing products
    similarity_scores = cosine_similarity(
        count_matrix, new_product_feature_vector)

    # Update the similarity matrix
    global cosine_sim
    cosine_sim = np.append(cosine_sim, similarity_scores, axis=1)

    # Save the updated DataFrame to the data-set file
    df.to_csv("data-set.csv", index=False)

    return "product has been added"
