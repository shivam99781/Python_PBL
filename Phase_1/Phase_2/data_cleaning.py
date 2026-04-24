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

    df[col] = np.where(df[col] < lower, lower,
              np.where(df[col] > upper, upper, df[col]))

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

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print(df[["property_type_Villa", "property_type_House"]].head())

#  CORRELATION HEATMAP

print(" GRAPH OF CORRELATION ")
plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)

sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



# REGRESSION MODEL
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

#  Separate data
X = df.drop("price", axis=1)
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Result
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))



#  GRAPH (ACTUAL vs PREDICTED)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()


# GRAPH  (ERROR GRAPH)

error = y_test - y_pred

plt.scatter(y_test, error)
plt.xlabel("Actual Price")
plt.ylabel("Error")
plt.title("Error Graph")
plt.show()
