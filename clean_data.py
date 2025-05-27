import pandas as pd
import re
import os

# Create 'data' directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load raw CSV
df = pd.read_csv("Real Estate Data V21.csv")

# Convert price strings to numeric lakhs
def convert_price_to_lakhs(price_str):
    if pd.isnull(price_str):
        return None
    price_str = price_str.replace("₹", "").strip()
    if "Cr" in price_str:
        return float(re.sub(r"[^\d.]", "", price_str)) * 100
    elif "L" in price_str:
        return float(re.sub(r"[^\d.]", "", price_str))
    else:
        return None

df["Price_Lakhs"] = df["Price"].apply(convert_price_to_lakhs)
df["Location"] = df["Location"].str.strip()
df = df.dropna(subset=["Price_Lakhs"])
df["City"] = df["Location"].apply(lambda x: x.split(",")[-1].strip() if "," in x else x)

# Save cleaned CSV
df.to_csv("data/real_estate.csv", index=False)
print("✅ Data cleaned and saved to data/real_estate.csv")
