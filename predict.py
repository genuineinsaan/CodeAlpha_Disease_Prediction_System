import joblib
import pandas as pd
import numpy as np

# Load Model and Encoder
model = joblib.load("model/disease_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")

# Load Dataset Columns
data = pd.read_csv("dataset/Training.csv")
feature_names = data.drop("prognosis", axis=1).columns


def predict_disease(symptoms):

    # Create input vector
    input_data = [0] * len(feature_names)

    for symptom in symptoms:

        if symptom in feature_names:

            index = list(feature_names).index(symptom)
            input_data[index] = 1

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    # Main Prediction
    prediction = model.predict(input_df)

    disease = encoder.inverse_transform(prediction)[0]

    # Confidence Score
    probabilities = model.predict_proba(input_df)[0]

    confidence = round(
        np.max(probabilities) * 100,
        2
    )

    # Top 3 Predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]

    top_predictions = []

    for index in top_indices:

        disease_name = encoder.inverse_transform([index])[0]

        probability = round(
            probabilities[index] * 100,
            2
        )

        top_predictions.append(
            {
                "disease": disease_name,
                "confidence": probability
            }
        )

    return (
        disease,
        confidence,
        top_predictions
    )