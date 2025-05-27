import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# Load cleaned data
df = pd.read_csv("data/real_estate.csv")

# Preprocess Balcony
df["Balcony"] = df["Balcony"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

# Features and target
X = df[["Total_Area", "Price_per_SQFT", "Baths", "Balcony", "City"]]
y = df["Price_Lakhs"]

# Categorical & numerical columns
cat_cols = ["City"]
num_cols = ["Total_Area", "Price_per_SQFT", "Baths", "Balcony"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

# ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/price_predictor.pkl")
print("✅ Model trained and saved to models/price_predictor.pkl")
