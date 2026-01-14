import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open("car_price_xgboost.pkl", "rb"))

st.header("Car Price Prediction Model")

cars_data = pd.read_csv("Cardetails.csv")

# Extract brand name
def get_brand_name(car_name):
    return car_name.split(" ")[0]

cars_data["name"] = cars_data["name"].apply(get_brand_name)

# Fix mileage
cars_data["mileage"] = cars_data["mileage"].astype(str).str.extract(r"([\d.]+)")
cars_data["mileage"] = pd.to_numeric(cars_data["mileage"], errors="coerce")

# Fix engine
cars_data["engine"] = (
    cars_data["engine"]
    .astype(str)
    .str.lower()
    .str.replace("cc", "", regex=False)
    .str.strip()
)
cars_data["engine"] = pd.to_numeric(cars_data["engine"], errors="coerce")

cars_data = cars_data.dropna(subset=["mileage", "engine"])

# ------------------ UI ------------------

CarName = st.selectbox("Select Car Brand", cars_data["name"].unique())
FuelType = st.selectbox("Select Fuel Type", cars_data["fuel"].unique())
SellerType = st.selectbox("Select Seller Type", cars_data["seller_type"].unique())
Transmission = st.selectbox("Select Transmission Type", cars_data["transmission"].unique())

ManufactureYear = st.slider(
    "Car Manufacture Year",
    int(cars_data["year"].min()),
    int(cars_data["year"].max()),
    2015
)

Mileage = st.slider(
    "Car Mileage (km/l)",
    float(cars_data["mileage"].min()),
    float(cars_data["mileage"].max()),
    24.5
)

KmsDriven = st.slider(
    "Number of Kms driven",
    int(cars_data["km_driven"].min()),
    int(cars_data["km_driven"].max()),
    50000
)

Engine = st.slider(
    "Engine Capacity (CC)",
    int(cars_data["engine"].min()),
    int(cars_data["engine"].max()),
    1200
)

Power = st.slider("Power (bhp)", 10, 350, 74)
owner = st.selectbox('Owner type', cars_data['owner'].unique())
seats = st.slider('No of Seats', 5,10)
 
# ------------------ Prediction ------------------

if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[CarName,ManufactureYear,KmsDriven,FuelType,SellerType,Transmission,owner,Mileage,Engine,Power,seats]],
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
    
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
                           [0,1,2,3,4], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[0,1,2,3], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[0,1,2], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
                          ,inplace=True)

    car_price = model.predict(input_data_model).astype(int)

    st.markdown('Car Price is going to be '+ str(car_price[0]))

    
    
    
