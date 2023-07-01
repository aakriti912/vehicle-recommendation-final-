def combine_features(row):
    vehicle_name = row["vehicle_name"]
    fuel_electric= row["fuel_electric"]
    vehicle_type= row["vehicle_type"]

    vehicleNameWeight = vehicle_name + " " + vehicle_name
  

    return f"{vehicleNameWeight} { fuel_electric } {vehicle_type}".lower()
