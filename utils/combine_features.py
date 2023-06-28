def combine_features(row):
    name = row["name"]
    categoryName = row["categoryName"].replace(" ", "_")
    subCategory = row["subCategory"].replace(" ", "_")
    brand = row["brand"]

    categoryWeight = categoryName + " " + categoryName
    subCategoryWeight = subCategory + " " + subCategory

    return f"{categoryWeight} { subCategoryWeight } {brand} {name}".lower()
