import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Use RandomForestRegressor for regression task
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Load dataset
df = pd.read_csv(r"C:\Cognifyz Technologies\Tasks\task 1 - predict restaurant ratings\Dataset .csv")  # make sure this file is in the same folder

# Convert 'Cuisines' string to list
df["Cuisines"] = df["Cuisines"].fillna("").apply(lambda x: [i.strip() for i in x.split(",")])

# Encode 'Cuisines' as multi-label
mlb = MultiLabelBinarizer()
cuisines_encoded = mlb.fit_transform(df["Cuisines"])

# Save MultiLabelBinarizer
joblib.dump(mlb, "mlb.pkl")

# Drop non-numeric or unnecessary columns
df = df.drop(["Restaurant ID", "Restaurant Name", "Cuisines"], axis=1)

# Encode binary categorical features (Yes/No)
binary_columns = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=[
    "Country Code", "City", "Address", "Locality", "Locality Verbose",
    "Currency", "Rating color", "Rating text"
])

# Append encoded cuisines to the dataframe
cuisine_df = pd.DataFrame(cuisines_encoded, columns=mlb.classes_)
df = pd.concat([df, cuisine_df], axis=1)

# Drop any remaining rows with NaN
df.dropna(inplace=True)

# Define X and y (now predicting 'Aggregate rating' instead of 'Price range')
X = df.drop("Aggregate rating", axis=1)  # Features
y = df["Aggregate rating"]  # Target variable

# Save column names for use during prediction
joblib.dump(X.columns.tolist(), "dataset_columns.pkl")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (use RandomForestRegressor for regression task)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully.")
