from fastapi import APIRouter
from fastapi import HTTPException, status
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.combine_features import combine_features

router = APIRouter()


count_matrix = None
cosine_sim = None


def calculate_cosine_similarity():
    global count_matrix, cosine_sim

    df = pd.read_csv("similarity-data-set.csv")

    features = ['brand']
    for feature in features:
        df[feature] = df[feature].fillna('')

    df["combined_features"] = df.apply(combine_features, axis=1)

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)


calculate_cosine_similarity()


@router.get("/{product_id}")
def get_similar_products(product_id: int):
    global cosine_sim

    df = pd.read_csv("similarity-data-set.csv")

    product_indices = df[df["id"] == product_id].index

    # if len(product_indices) == 0:
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND, detail="product not found")

    if len(product_indices) == 0:
        return []

    product_index = product_indices[0]

    similar_products = list(enumerate(cosine_sim[product_index]))
    sorted_similar_products = sorted(
        similar_products, key=lambda x: x[1], reverse=True)

    result = [{"productId": int(df.iloc[item[0]]["id"]), "similarity": item[1]}
              for item in sorted_similar_products if item[1] >= 0.2 and int(df.iloc[item[0]]["id"]) != product_id]

    # result = [int(df.iloc[item[0]]["id"]) for item in sorted_similar_products if item[1]
    #           > 0 and int(df.iloc[item[0]]["id"]) != product_id]

    return result[0:10]


@router.post("/products")
def add_product(product_data: dict):
    df = pd.read_csv("similarity-data-set.csv")

    product_id = product_data["id"]
    product_exists = False

    if product_id in df["id"].values:
        product_exists = True
        product_index = df[df["id"] == product_id].index[0]
        df = df.drop(product_index)

    df.loc[len(df)] = product_data
    df.to_csv("similarity-data-set.csv", index=False)

    calculate_cosine_similarity()

    return {"message": f"the product has been { 'updated' if product_exists else 'added' }"}


@router.delete("/products/{product_id}")
def remove_product(product_id: int):
    df = pd.read_csv("similarity-data-set.csv")

    if product_id not in df["id"].values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="product not found")

    product_index = df[df["id"] == product_id].index[0]

    df = df.drop(product_index)
    df.to_csv("similarity-data-set.csv", index=False)

    calculate_cosine_similarity()

    return {"message": "the product has been removed"}
