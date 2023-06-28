from fastapi import APIRouter
from fastapi import HTTPException, status
import pandas as pd
import httpx


router = APIRouter()


df = pd.read_csv("recommendation-data-set.csv")


@router.get("/{user_id}")
async def get_product_recommendations(user_id: int):
    global df

    user_orders = df[df["userId"] == user_id]["id"].drop_duplicates().to_list()

    if len(user_orders) == 0:
        return []

    recommendations = []

    for id in user_orders:
        similar_products = []

        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:8080/similar/{id}")
            similar_products = response.json()
            recommendations.extend(similar_products)

    recommendations = sorted(
        recommendations, key=lambda x: x["similarity"], reverse=True)

    unique_product_ids = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec["productId"] not in unique_product_ids and rec["productId"] not in user_orders:
            unique_recommendations.append(rec)
            unique_product_ids.add(rec["productId"])

    return unique_recommendations


@router.post("/orders")
def add_user_order(order_data: dict):
    global df

    order_id = order_data["orderId"]

    if order_id in df["orderId"].values:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="the order already exists")

    df.loc[len(df)] = order_data
    df.to_csv("recommendation-data-set.csv", index=False)

    return {"message": "the order has been added"}
