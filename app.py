import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ------------------ Load Data & Train Models ------------------
@st.cache_data
def load_and_train_rf():
    # Load cleaned dataset
    df = pd.read_csv("weather_cleaned.csv")

    # City columns
    cities = ["Lucknow", "Delhi", "Kanpur", "Mumbai", "Noida"]

    # Convert all city columns to numeric (safety step)
    for city in cities:
        df[city] = pd.to_numeric(df[city], errors="coerce")

    # Fill missing values (if any)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # Create numeric month column (1–12)
    df["Month_Num"] = range(1, len(df) + 1)

    # Train Random Forest model for each city & compute R²
    models = {}
    r2_scores = {}
    X = df[["Month_Num"]]

    for city in cities:
        y = df[city]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        model.fit(X, y)
        models[city] = model

        # Predict on training data to get R² score
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        r2_scores[city] = r2

    return df, models, cities, r2_scores

# Load data and trained models
df, models, cities, r2_scores = load_and_train_rf()

# ------------------ Streamlit User Interface ------------------
st.title("🌦 Weather Temperature Prediction")
st.write(
    "This app predicts **monthly average temperature** using historical data "
    "and a Random Forest Regression model."
)

# City selection
selected_city = st.selectbox("✅ Select a City:", cities)

# Show R² accuracy for selected city
st.write(
    f"📈 Model R² Score for **{selected_city}**: "
    f"**{r2_scores[selected_city]:.3f}**"
)

# Month selection
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

selected_month_num = st.slider("✅ Select Month (1–12):", 1, 12)
selected_month_name = month_names[selected_month_num]

st.write(f"📅 Selected Month: **{selected_month_name}**")

# Optional: Show historical data
with st.expander("📊 Show Historical Temperature Data"):
    st.line_chart(df.set_index("Month_Num")[selected_city])

# ------------------ Prediction Button ------------------
if st.button("🔍 Predict Temperature"):
    X_input = pd.DataFrame({"Month_Num": [selected_month_num]})
    prediction = models[selected_city].predict(X_input)[0]

    st.success(
        f"Predicted temperature for **{selected_city}** in **{selected_month_name}** "
        f"is **{prediction:.2f} °C**"
    )
