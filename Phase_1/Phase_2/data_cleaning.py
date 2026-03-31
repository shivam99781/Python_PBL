import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("house_price_dataset_5000.csv")
print(df)
print(df.columns)

#for outliers...
print("**********outliers**********")
q1 = df['property_area_sqft'].quantile(0.25)
q3 = df['property_area_sqft'].quantile(0.75)
IQR = q3 - q1
min = q1 - 1.5 * IQR
max = q3 + 1.5 * IQR

outliers = (df['property_area_sqft'] < min) | (df['property_area_sqft'] > max)
print(outliers)


#normalization for convert 0 to 1 range...
print("********** AFTER NORMALIZATION **********") 
scaler = MinMaxScaler()
df[['property_area_sqft', 'bedrooms',  'bathrooms', 'floors', 'property_age', 'renovation_status', 'lot_size', 'distance_to_city_km','neighborhood_quality_score','construction_quality_rating' ,'energy_efficiency_score','water_supply_reliability' ,'electricity_supply_reliability','internet_availability_score' ,'green_space_index', 'flood_risk_index','noise_pollution_level']] = scaler.fit_transform(df[['property_area_sqft', 'bedrooms',  'bathrooms', 'floors', 'property_age', 'renovation_status', 'lot_size', 'distance_to_city_km','neighborhood_quality_score','construction_quality_rating' ,'energy_efficiency_score','water_supply_reliability','electricity_supply_reliability','internet_availability_score' ,'green_space_index','flood_risk_index','noise_pollution_level']])
print(df)

#ONE HOT ENCODING TO CONVERT CATAGORICAL TO NUMERICAL FORM...
print("********** AFTER ONE HOT ENCODING **********")
df = pd.get_dummies(df, columns=['property_type'], drop_first=True)
print(df)

# TO DROP ONE EXTRA COLUMN BY USING .drop....
print("********** AFTER DROP ONE COLUMN ********** ")
df.drop("property_type_House", axis=1, inplace=True)
print(df)

# property_type_villa IS IN BOOLEAN CONVERT IT IN THE RANGE OF 0-1...
print(" ********** AFTER CONVERT BOOLEAN TO RANGE OF 0-1 ********** ")
df["property_type_Villa"] = df["property_type_Villa"].astype(int)
print(df["property_type_Villa"])


