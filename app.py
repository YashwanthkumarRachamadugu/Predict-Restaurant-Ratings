import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load(r"C:\Cognifyz Technologies\Tasks\task 1 - predict restaurant ratings\model.pkl")
mlb = joblib.load(r"C:\Cognifyz Technologies\Tasks\task 1 - predict restaurant ratings\mlb.pkl")
columns = joblib.load(r"C:\Cognifyz Technologies\Tasks\task 1 - predict restaurant ratings\dataset_columns.pkl")

st.title("🍽️ Restaurant Aggregate Rating Predictor")
st.markdown("Fill in the details below to predict the aggregate rating of a restaurant.")

# User input fields
has_table_booking = st.selectbox("Has Table Booking?", ["Yes", "No"])
has_online_delivery = st.selectbox("Has Online Delivery?", ["Yes", "No"])
is_delivering_now = st.selectbox("Is Delivering Now?", ["Yes", "No"])
switch_to_order_menu = st.selectbox("Switch to Order Menu?", ["Yes", "No"])

# Rating prediction inputs (based on dataset)
votes = st.number_input("Number of Votes", min_value=0, step=1)
country_code = st.selectbox("Country Code", ["1", "14", "30", "37", "94", "148", "162", "166"])  # Add real options
currency = st.selectbox("Currency", ["Botswana Pula(P)", "Brazilian Real(R$)", "Dollar($)", "Emirati Diram(AED)", "Indian Rupees(Rs.)", "Pounds(£)", "Qatari Rial(QR)", "Rand(R)", "Sri Lankan Rupee(LKR)", "Turkish Lira(TL)"])  # Add full list
rating_color = st.selectbox("Rating Color", ["Dark Green", "Green", "Yellow", "Orange", "Red", "White"])
rating_text = st.selectbox("Rating Text", ["Excellent", "Very Good", "Good", "Average", "Poor", "Not rated"])

# Cuisine selection
selected_cuisines = st.multiselect("Select Cuisines", mlb.classes_)

# Convert user inputs
binary_map = {"Yes": 1, "No": 0}
input_dict = {
    "Has Table booking": binary_map[has_table_booking],
    "Has Online delivery": binary_map[has_online_delivery],
    "Is delivering now": binary_map[is_delivering_now],
    "Switch to order menu": binary_map[switch_to_order_menu],
    "Votes": votes,
    f"Country Code_{country_code}": 1,
    f"Currency_{currency}": 1,
    f"Rating color_{rating_color}": 1,
    f"Rating text_{rating_text}": 1,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Encode cuisines
cuisine_encoded = mlb.transform([selected_cuisines])
cuisine_df = pd.DataFrame(cuisine_encoded, columns=mlb.classes_)

# Combine input (step 1: combine input_df and cuisine_df)
input_full = pd.concat([input_df, cuisine_df], axis=1)

# Step 2: Add missing columns (instead of looping, create a DataFrame with missing columns)
missing_cols = [col for col in columns if col not in input_full.columns]
missing_df = pd.DataFrame(0, index=input_full.index, columns=missing_cols)

# Step 3: Concatenate the missing columns to the input_full DataFrame
input_full = pd.concat([input_full, missing_df], axis=1)

# Step 4: Reorder columns to match the training data
input_full = input_full[columns]

# Predict
if st.button("Predict Aggregate Rating"):
    prediction = model.predict(input_full)
    st.success(f"🔮 Predicted Aggregate Rating: {prediction[0]:.2f}")
