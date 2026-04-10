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

#  HANDLE NULL VALUES

print("\nNULL VALUES BEFORE:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nNULL VALUES AFTER:")
print(df.isnull().sum())

#  HANDLE 0 VALUES

print("\nHANDLING ZERO VALUES")

cols = ['bedrooms', 'bathrooms', 'floors']
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].mean())

#  OUTLIER REMOVAL 

print("\nREMOVING OUTLIERS")

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

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("SHAPE AFTER OUTLIER REMOVAL:", df.shape)

#  NORMALIZATION

print("\nNORMALIZATION")

scaler = MinMaxScaler()

columns = [
    'property_area_sqft', 'bedrooms', 'bathrooms', 'floors',
    'property_age', 'lot_size', 'distance_to_city_km',
    'neighborhood_quality_score', 'construction_quality_rating',
    'energy_efficiency_score', 'water_supply_reliability',
    'electricity_supply_reliability', 'internet_availability_score',
    'green_space_index', 'flood_risk_index', 'noise_pollution_level'
]

df[columns] = scaler.fit_transform(df[columns])

print(df[columns].head())

#  ONE HOT ENCODING

print("\nONE HOT ENCODING")

df = pd.get_dummies(df, columns=['property_type'], drop_first=True)

#  BOOLEAN TO INT

print("\nBOOLEAN TO INT")

df["property_type_Villa"] = df["property_type_Villa"].astype(int)
df["property_type_House"] = df["property_type_House"].astype(int)

print(df[["property_type_Villa", "property_type_House"]].head())

#  CORRELATION HEATMAP

print(" GRAPH OF CORRELATION ")

plt.figure()
corr = df.corr()

sns.heatmap(corr, annot=False)
plt.title("Correlation Heatmap")
plt.show()
