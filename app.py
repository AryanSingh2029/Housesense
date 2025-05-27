import streamlit as st
st.set_page_config(page_title="🏡 Real Estate Dashboard", layout="wide")

import pandas as pd
import plotly.express as px
import joblib

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/real_estate.csv")
    df["City"] = df["Location"].apply(lambda x: x.split(",")[-1].strip() if "," in x else x)
    return df

df = load_data()

st.title("📊 Indian Real Estate Insights")

# Tabs
tabs = st.tabs([
    "📍 Price Heatmap",
    "📈 Price Predictor",
    "💰 ROI Estimator",
    "🔍 Property Recommender",
    "🏷️ Property Classifier"
])

# ------------------ Tab 1: Price Heatmap ------------------ #
with tabs[0]:
    st.header("📍 City-wise Price Heatmap")
    st.write("🔄 Visualize average price, price per sq.ft., and property area by city.")

    # Average price by city
    avg_price_city = df.groupby("City")["Price_Lakhs"].mean().reset_index().sort_values(by="Price_Lakhs", ascending=False)
    fig1 = px.bar(avg_price_city, x="City", y="Price_Lakhs", title="Average Property Price (Lakhs) by City", color="Price_Lakhs")
    st.plotly_chart(fig1, use_container_width=True)

    # Avg price per sqft by city
    avg_sqft_price = df.groupby("City")["Price_per_SQFT"].mean().reset_index().sort_values(by="Price_per_SQFT", ascending=False)
    fig2 = px.bar(avg_sqft_price, x="City", y="Price_per_SQFT", title="Average Price per Sq.Ft by City", color="Price_per_SQFT")
    st.plotly_chart(fig2, use_container_width=True)

    # Area distribution
    fig3 = px.box(df, x="City", y="Total_Area", title="Distribution of Property Areas by City")
    st.plotly_chart(fig3, use_container_width=True)

# ------------------ Tab 2: Price Predictor ------------------ #
with tabs[1]:
    st.header("📈 Predict Property Price")
    st.write("🧠 Enter property details to estimate its price.")

    # Load model
    model = joblib.load("models/price_predictor.pkl")

    # Input form
    with st.form("predict_form"):
        total_area = st.number_input("Total Area (sq. ft.)", min_value=100, max_value=10000, value=1000)
        price_per_sqft = st.number_input("Price per Sq.Ft (INR)", min_value=1000, max_value=10000, value=5000)
        baths = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])
        balcony = st.selectbox("Balcony", ["Yes", "No"])
        city = st.selectbox("City", sorted(df["City"].unique()))
        
        submit = st.form_submit_button("Predict Price 💰")

    # Predict on submission
    if submit:
        balcony_num = 1 if balcony == "Yes" else 0
        input_df = pd.DataFrame([{
            "Total_Area": total_area,
            "Price_per_SQFT": price_per_sqft,
            "Baths": baths,
            "Balcony": balcony_num,
            "City": city
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"🏷️ Estimated Property Price: ₹ {round(prediction, 2)} Lakhs")

# ------------------ Placeholder Tabs ------------------ #
with tabs[2]:
    st.header("💰 ROI Estimator for Investors")
    st.write("📊 Estimate your return on investment based on rent and property price.")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Total Area (sq. ft.)", min_value=100, max_value=10000, value=1000)
        price = st.number_input("Purchase Price (in ₹ Lakhs)", min_value=10, max_value=1000, value=50)
    with col2:
        city = st.selectbox("City", sorted(df["City"].unique()))
        rent_per_sqft = st.slider("Estimated Rent per Sq.Ft (₹)", min_value=5, max_value=100, value=25)

    if st.button("Estimate ROI 📈"):
        monthly_rent = rent_per_sqft * area
        annual_rent = monthly_rent * 12
        maintenance = area * 15  # ₹15/sqft annually
        roi = ((annual_rent - maintenance) / (price * 100000)) * 100

        st.metric("🏠 Monthly Rent", f"₹{monthly_rent:,.0f}")
        st.metric("📅 Annual Rent", f"₹{annual_rent:,.0f}")
        st.metric("🧾 Estimated ROI", f"{roi:.2f}%")
with tabs[3]:
    st.header("🔍 Personalized Property Recommender")
    st.write("🎯 Get a list of properties that match your needs.")

    with st.form("recommend_form"):
        max_budget = st.slider("Your Budget (₹ Lakhs)", min_value=10, max_value=500, value=100)
        selected_city = st.selectbox("Preferred City", sorted(df["City"].unique()))
        min_area = st.number_input("Minimum Area (sq.ft)", min_value=200, max_value=10000, value=800)
        min_baths = st.selectbox("Minimum Number of Bathrooms", [1, 2, 3, 4, 5])
        sort_by = st.selectbox("Sort Results By", ["Price (Low to High)", "Price (High to Low)", "Area (Small to Large)", "Area (Large to Small)"])
        submit = st.form_submit_button("Find Properties 🔍")

    if submit:
        filtered = df[
            (df["Price_Lakhs"] <= max_budget) &
            (df["City"] == selected_city) &
            (df["Total_Area"] >= min_area) &
            (df["Baths"] >= min_baths)
        ]

        if filtered.empty:
            st.warning("❌ No matching properties found. Try adjusting your filters.")
        else:
            if sort_by == "Price (Low to High)":
                filtered = filtered.sort_values(by="Price_Lakhs")
            elif sort_by == "Price (High to Low)":
                filtered = filtered.sort_values(by="Price_Lakhs", ascending=False)
            elif sort_by == "Area (Small to Large)":
                filtered = filtered.sort_values(by="Total_Area")
            elif sort_by == "Area (Large to Small)":
                filtered = filtered.sort_values(by="Total_Area", ascending=False)

            st.success(f"✅ Found {len(filtered)} matching properties:")
            st.dataframe(filtered[["Name", "Location", "Price_Lakhs", "Total_Area", "Baths", "Balcony"]])
with tabs[4]:
    st.header("🏷️ Luxury vs Budget Classifier")
    st.write("📌 Classify a property as 'Luxury' or 'Budget' based on its features.")

    # Load classifier model
    clf_model = joblib.load("models/classifier.pkl")

    # Input form
    with st.form("classify_form"):
        area = st.number_input("Total Area (sq. ft.)", min_value=200, max_value=10000, value=1000)
        price_sqft = st.number_input("Price per Sq.Ft (₹)", min_value=1000, max_value=20000, value=6000)
        baths = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])
        balcony = st.selectbox("Balcony", ["Yes", "No"])
        city = st.selectbox("City", sorted(df["City"].unique()))
        submit = st.form_submit_button("Classify 🏷️")

    if submit:
        balcony_num = 1 if balcony == "Yes" else 0
        input_df = pd.DataFrame([{
            "Total_Area": area,
            "Price_per_SQFT": price_sqft,
            "Baths": baths,
            "Balcony": balcony_num,
            "City": city
        }])

        label = clf_model.predict(input_df)[0]
        st.success(f"🏠 This property is classified as: **{label}**")
