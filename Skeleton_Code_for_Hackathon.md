"""
Skeleton: Airbnb Equity Engine API
Goal: Predict fairness of Airbnb listings and offer smart recommendations.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

app = Flask(__name__)

# === Load Data (participants need to provide 'airbnb.csv') ===
df = pd.read_csv("airbnb.csv")

# === TODO: Preprocessing Steps ===
# - Drop rows with missing key fields (e.g., Price, Beds)
# - Clip unrealistic values for Price and Beds
# - Define "Affordable", "Accessible", and "Fair_Listing" based on logic

# Example placeholder for preprocessing
# df["Affordable"] = ...
# df["Fair_Listing"] = ...

# === TODO: Encode Categorical Features ===
# Use LabelEncoder for 'Room Type' and 'Property Type'

# === TODO: Train Logistic Regression Model ===
# Split your data into features and target
# Train a classification model to predict fair listings

# === TODO: Calculate Fairness Score Per Neighborhood ===
# Group by 'Neighbourhood' and compute fairness score using custom weights

@app.route("/api/fairness-scores", methods=["GET"])
def get_fairness_scores():
    # TODO: Return sorted fairness scores
    return jsonify({"message": "Fairness scores placeholder"})

@app.route("/api/listing-fairness", methods=["POST"])
def check_listing_fairness():
    content = request.json
    # TODO: Predict whether a listing is fair or not using trained model
    return jsonify({"predicted_fair_listing": "TBD"})

@app.route("/api/recommend-fair-listings", methods=["GET"])
def recommend_fair():
    # TODO: Return 10 random fair listings
    return jsonify({"recommendations": []})

@app.route("/api/ai-suggestion", methods=["POST"])
def smart_suggestion():
    content = request.json
    # TODO: Generate recommendations on how to make a listing more fair
    return jsonify({
        "predicted_fair": "TBD",
        "smart_tip": "Add tips here based on rules"
    })

if __name__ == "__main__":
    app.run(debug=True)
