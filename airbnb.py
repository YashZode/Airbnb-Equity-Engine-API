"""
Backend API for Airbnb Equity Engine
This Flask application exposes endpoints to:
- Get fairness scores per neighborhood
- Predict listing fairness using ML
- Recommend fair listings
- Provide AI-based smart tips using ML + LLM-style logic
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv("airbnb.csv")
df.dropna(subset=["Neighbourhood", "Price", "Room Type", "Beds", "Review Scores Rating"], inplace=True)
df["Price"] = df["Price"].clip(0, 1000)
df["Beds"] = df["Beds"].clip(0, 10)

# Threshold for affordability
afford_threshold = df["Price"].quantile(0.25)
df["Accessible"] = ((df["Beds"] >= 2) & (df["Room Type"] != "Shared room")).astype(int)
df["Affordable"] = (df["Price"] <= afford_threshold).astype(int)
df["Fair_Listing"] = ((df["Affordable"] == 1) & (df["Accessible"] == 1)).astype(int)

# Encode features for ML
le_room = LabelEncoder()
le_prop = LabelEncoder()
df["Room_Type_Enc"] = le_room.fit_transform(df["Room Type"])
df["Property_Type_Enc"] = le_prop.fit_transform(df["Property Type"])

X = df[["Price", "Beds", "Room_Type_Enc", "Property_Type_Enc", "Review Scores Rating"]]
y = df["Fair_Listing"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Neighborhood stats
neighborhood_stats = df.groupby("Neighbourhood").agg(
    Listings=("Name", "count"),
    Avg_Price=("Price", "mean"),
    Fair_Listings=("Fair_Listing", "sum"),
    Accessible_Listings=("Accessible", "sum"),
    Avg_Rating=("Review Scores Rating", "mean")
)
neighborhood_stats["Fairness_Score"] = (
    (neighborhood_stats["Fair_Listings"] / neighborhood_stats["Listings"]) * 0.5 +
    (neighborhood_stats["Accessible_Listings"] / neighborhood_stats["Listings"]) * 0.3 +
    (neighborhood_stats["Avg_Rating"] / 100) * 0.2
) * 100

@app.route("/api/fairness-scores", methods=["GET"])
def get_fairness_scores():
    data = neighborhood_stats.sort_values("Fairness_Score", ascending=False).reset_index()
    return jsonify(data.to_dict(orient="records"))

@app.route("/api/listing-fairness", methods=["POST"])
def check_listing_fairness():
    content = request.json
    try:
        price = float(content.get("price"))
        beds = float(content.get("beds"))
        room_type = content.get("room_type", "Private room")
        prop_type = content.get("property_type", "Apartment")
        rating = float(content.get("review_score", 90))

        rt_enc = le_room.transform([room_type])[0]
        pt_enc = le_prop.transform([prop_type])[0]

        X_input = pd.DataFrame([[price, beds, rt_enc, pt_enc, rating]], columns=["Price", "Beds", "Room_Type_Enc", "Property_Type_Enc", "Review Scores Rating"])
        prediction = clf.predict(X_input)[0]

        return jsonify({"predicted_fair_listing": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/recommend-fair-listings", methods=["GET"])
def recommend_fair():
    listings = df[df["Fair_Listing"] == 1].sample(10).reset_index(drop=True)
    return jsonify(listings[["Name", "Neighbourhood", "Price", "Beds", "Room Type"]].to_dict(orient="records"))

@app.route("/api/ai-suggestion", methods=["POST"])
def smart_suggestion():
    content = request.json
    try:
        price = float(content.get("price"))
        beds = float(content.get("beds"))
        room_type = content.get("room_type", "Private room")
        prop_type = content.get("property_type", "Apartment")
        rating = float(content.get("review_score", 90))

        rt_enc = le_room.transform([room_type])[0]
        pt_enc = le_prop.transform([prop_type])[0]

        X_input = pd.DataFrame([[price, beds, rt_enc, pt_enc, rating]], columns=["Price", "Beds", "Room_Type_Enc", "Property_Type_Enc", "Review Scores Rating"])
        prediction = clf.predict(X_input)[0]

        if prediction == 1:
            tip = "Great job! Your listing meets fairness criteria. Keep it up!"
        else:
            tips = []
            if price > afford_threshold:
                tips.append(f"Consider reducing your price from ${price:.0f} to under ${afford_threshold:.0f}.")
            if beds < 2:
                tips.append("Adding more sleeping space (e.g., extra bed) could help.")
            if room_type == "Shared room":
                tips.append("Consider switching from 'Shared room' to 'Private room' or 'Entire home'.")
            if rating < 90:
                tips.append("Improve your review rating by enhancing cleanliness and guest experience.")
            tip = " ".join(tips)

        return jsonify({"predicted_fair": int(prediction), "smart_tip": tip})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
