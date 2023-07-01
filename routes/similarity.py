from fastapi import APIRouter
from fastapi import HTTPException, status
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.combine_features import combine_features

router = APIRouter()


count_matrix = None
cosine_sim = None

# .\pratikenv\Scripts\activate

def calculate_cosine_similarity():
    global count_matrix, cosine_sim

    df = pd.read_csv("rentedwheels_vehicle1.csv")


    features = ['vehicle_type']
    for feature in features:
        df[feature] = df[feature].fillna('')

    df["combined_features"] = df.apply(combine_features, axis=1)

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)


calculate_cosine_similarity()


@router.get("/{vehicle_id}")
def get_similar_vehicles(vehicle_id: int):
    global cosine_sim

    df = pd.read_csv("rentedwheels_vehicle1.csv")

    vehicle_indices = df[df["id"] == vehicle_id].index
# http://localhost:8080/similar/77
    # if len(product_indices) == 0:
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND, detail="product not found")

    if len(vehicle_indices) == 0:
        return []

    vehicle_index = vehicle_indices[0]

    similar_vehicles = list(enumerate(cosine_sim[vehicle_index]))
    sorted_similar_vehicles = sorted(
        similar_vehicles, key=lambda x: x[1], reverse=True)

    result = [{"vehicleId": int(df.iloc[item[0]]["id"]), "similarity": item[1]}
              for item in sorted_similar_vehicles if item[1] > 0 and int(df.iloc[item[0]]["id"]) != vehicle_id]

    # result = [int(df.iloc[item[0]]["id"]) for item in sorted_similar_products if item[1]
    #           > 0 and int(df.iloc[item[0]]["id"]) != product_id]

    return result[0:10]


@router.post("/vehicles")
def add_vehicle(vehicle_data: dict):
    df = pd.read_csv("rentedwheels_vehicle1.csv")

    vehicle_id = vehicle_data["id"]  
    #  \\backend bata aaune similar product
    vehicle_exists = False

    if vehicle_id in df["id"].values:
        vehicle_exists = True
        vehicle_index = df[df["id"] == id].index[0]
        df = df.drop(vehicle_index)

    df.loc[len(df)] = vehicle_data
    df.to_csv("rentedwheels_vehicle1.csv", index=False)

    calculate_cosine_similarity()

    return {"message": f"the product has been { 'updated' if product_exists else 'added' }"}


@router.delete("/vehicles/{vehicle_id}")
def remove_product(vehicle_id: int):
    df = pd.read_csv("rentedwheels_vehicle1.csv")

    if product_id not in df["id"].values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="vehicle not found")

    vehicle_index = df[df["id"] == vehicle_id].index[0]

    df = df.drop(vehicle_index)
    df.to_csv("rentedwheels_vehicle1.csv", index=False)

    calculate_cosine_similarity()

    return {"message": "the vehicle has been removed"}
