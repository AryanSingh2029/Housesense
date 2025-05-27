# 🏠 HouseSense: Real Estate Investment Analytics Dashboard

🌐 **Live App**: [https://housesense-gpjbtj5qcctd9kmvjp5qmj.streamlit.app](https://housesense-gpjbtj5qcctd9kmvjp5qmj.streamlit.app)

---

## 📌 Overview

**HouseSense** is an ML-powered real estate analytics dashboard that helps users explore, evaluate, and invest in Indian properties. It features data visualizations, predictive models, investment ROI estimators, and classification tools — all built using real-world housing data.

---

## 🧩 Features & How They Were Built

### 📍 City-wise Price Heatmap
> **Goal**: Understand city-level price and area trends

✅ **What I Did**:
- Extracted city names from the `Location` field
- Used **Plotly Express** to create:
  - **Bar plots** for average price and price per square foot by city
  - **Box plots** to show area distribution per city
- Implemented inside `tabs[0]` using `groupby` and `px.bar` / `px.box`

---

### 📈 Price Predictor
> **Goal**: Predict property prices based on features

✅ **What I Did**:
- Trained a **Random Forest Regressor** with:
  - Features: `Total_Area`, `Price_per_SQFT`, `Baths`, `Balcony`, `City`
- Encoded categorical `City` using `OneHotEncoder` inside a `Pipeline`
- Saved the model using `joblib` and loaded it in Streamlit
- User inputs collected via `st.form()` and predictions shown via `st.success()`

---

### 💰 ROI Estimator
> **Goal**: Help investors calculate return on property investments

✅ **What I Did**:
- Created a formula:
  ```python
  ROI = ((Annual Rent - Maintenance) / Purchase Price) * 100
Inputs:

Purchase price

Area

Rent per sq.ft.

Assumed fixed maintenance cost (₹15/sq.ft)

Used st.metric() to display results with a business-style UX

🔍 Property Recommender
Goal: Recommend properties based on user’s preferences

✅ What I Did:

Used filters:

Budget (in Lakhs)

City

Minimum Area

Number of Bathrooms

Added sorting options: by price or area, both ascending and descending

Displayed results using st.dataframe() with a clean tabular layout

🏷️ Luxury vs Budget Classifier
Goal: Classify a property as “Luxury” or “Budget”

✅ What I Did:

Defined a label rule:

Luxury: If area > 2000 sq.ft. or price/sq.ft > ₹7500

Trained a Random Forest Classifier

Used the same features as the predictor tab

User input collected via form, output shown via classification result

🛠️ Tech Stack
Frontend: Streamlit

Data Handling: Pandas

Machine Learning: Scikit-learn, Joblib

Visualizations: Plotly Express

Model Pipeline: OneHotEncoder + RandomForest (via Pipeline)

