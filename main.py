from typing import Union
from fastapi import FastAPI
from fastapi import HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def combined_features(row):
    name = row["name"]
    categoryName = row["categoryName"].replace(" ", "_")
    subCategory = row["subCategory"]
    brand = row["brand"]

    return f"{ categoryName } { categoryName } { subCategory } { subCategory } { brand } { name }".lower()


@app.get("/recommendations/{product_id}")
def get_recommendations(product_id: int):
    df = pd.read_csv("data-set.csv")
    product_indices = df[df["id"] == product_id].index

    if len(product_indices) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="product not found")

    product_index = product_indices[0]

    features = ['brand']
    for feature in features:
        df[feature] = df[feature].fillna('')

    df["combined_features"] = df.apply(combined_features, axis=1)

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)

    similar_products = list(enumerate(cosine_sim[product_index]))
    sorted_similar_products = sorted(
        similar_products, key=lambda x: x[1], reverse=True)

    result = [{"productId": int(df.iloc[item[0]]["id"]), "similarity": item[1]}
              for item in sorted_similar_products if item[1] > 0 and int(df.iloc[item[0]]["id"]) != product_id]

    return result[0:10]


@app.post("/products")
def add_product(product_data: dict):
    df = pd.read_csv("data-set.csv")
    product_id = product_data["id"]
    product_exists = False

    if product_id in df["id"].values:
        product_exists = True
        product_index = df[df["id"] == product_id].index[0]
        df = df.drop(product_index)

    df.loc[len(df)] = product_data
    df.to_csv("data-set.csv", index=False)

    return {"message": f"the product has been { 'updated' if product_exists else 'added' }"}


@app.delete("/products/{product_id}")
def remove_product(product_id: int):
    df = pd.read_csv("data-set.csv")

    if product_id not in df["id"].values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="product not found")

    product_index = df[df["id"] == product_id].index[0]

    df = df.drop(product_index)
    df.to_csv("data-set.csv", index=False)

    return {"message": "the product has been removed"}
