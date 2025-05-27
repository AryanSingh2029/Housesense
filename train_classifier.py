import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# Load cleaned data
df = pd.read_csv("data/real_estate.csv")

# Preprocess
df["Balcony"] = df["Balcony"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

# Define label: mark as Luxury if price/sq.ft > 7500 or area > 2000
df["Label"] = df.apply(lambda row: "Luxury" if row["Price_per_SQFT"] > 7500 or row["Total_Area"] > 2000 else "Budget", axis=1)

# Features and target
X = df[["Total_Area", "Price_per_SQFT", "Baths", "Balcony", "City"]]
y = df["Label"]

# Pipeline
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown='ignore'), ["City"])
], remainder='passthrough')

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/classifier.pkl")
print("✅ Classifier model saved to models/classifier.pkl")
