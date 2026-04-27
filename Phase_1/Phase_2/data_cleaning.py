print("*****************************PREPROCESSING****************************************")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("house_price_dataset_5000.csv")

print("DATASET PREVIEW:")
print(df.head())

print("\nCOLUMNS:")
print(df.columns)

# HANDLE NULL VALUES
df.fillna(df.mean(numeric_only=True), inplace=True)

# HANDLE 0 VALUES
cols_zero = ['bedrooms', 'bathrooms', 'floors']
df[cols_zero] = df[cols_zero].replace(0, np.nan)
df[cols_zero] = df[cols_zero].fillna(df[cols_zero].mean())

# OUTLIER REMOVAL
outlier_cols = [
    'property_area_sqft', 'lot_size', 'distance_to_city_km',
    'neighborhood_quality_score', 'construction_quality_rating'
]

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower,
              np.where(df[col] > upper, upper, df[col]))

# NORMALIZATION
cols = [
    'property_area_sqft','bedrooms','bathrooms','floors',
    'property_age','lot_size','distance_to_city_km',
    'neighborhood_quality_score','construction_quality_rating',
    'energy_efficiency_score','water_supply_reliability',
    'electricity_supply_reliability','internet_availability_score',
    'green_space_index','flood_risk_index','noise_pollution_level'
]

scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

# ONE HOT ENCODING
df = pd.get_dummies(df, columns=['property_type'], drop_first=True)

cols.append('property_type_Villa')

#  CORRELATION HEATMAP

print(" GRAPH OF CORRELATION ")
plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)

sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("*****************************************TRAINING ND TESTING************************************************")

# MODEL TRAINING
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X = df[cols]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMODEL PERFORMANCE:-")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# GRAPH 1
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

# GRAPH 2
error = y_test - y_pred
plt.scatter(y_test, error)
plt.xlabel("Actual Price")
plt.ylabel("Error")
plt.title("Error Graph")
plt.show()

print("******************************USER INPUT FOR TESTING THE PREDICTIONS ON ANOTHER INPUTs****************************")

print("\nEnter House Details:")

area = float(input("property_area_sqft: "))
bed = int(input("bedrooms: "))
bath = int(input("bathrooms: "))
floor = int(input("floors: "))
age = float(input("property_age: "))
lot = float(input("lot_size: "))
dist = float(input("distance_to_city_km: "))
neigh = float(input("neighborhood_quality_score: "))
const = float(input("construction_quality_rating: "))
energy = float(input("energy_efficiency_score: "))
water = float(input("water_supply_reliability: "))
elec = float(input("electricity_supply_reliability: "))
internet = float(input("internet_availability_score: "))
green = float(input("green_space_index: "))
flood = float(input("flood_risk_index: "))
noise = float(input("noise_pollution_level: "))

ptype = input("property_type (Villa/House): ")

# CREATE INPUT
input_list = [[area, bed, bath, floor, age, lot, dist, neigh, const,
               energy, water, elec, internet, green, flood, noise]]

# CONVERT TO DATAFRAME
input_df = pd.DataFrame(input_list, columns=cols[:-1])  

# SCALE INPUT
input_df = pd.DataFrame(scaler.transform(input_df), columns=cols[:-1])

# ADD PROPERTY TYPE
input_df['property_type_Villa'] = 1 if ptype.lower()=="villa" else 0


# PREDICT
price = model.predict(input_df)

print("After Prediction")
print("\n Predicted Price:", float(price[0]))
