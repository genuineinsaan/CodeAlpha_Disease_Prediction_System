from flask import Flask, render_template, request
from predict import predict_disease
import pandas as pd

app = Flask(__name__)

# Load symptom list
data = pd.read_csv("dataset/Training.csv")
symptoms_list = data.drop("prognosis", axis=1).columns.tolist()


@app.route("/")
def home():

    return render_template(
        "index.html",
        symptoms=symptoms_list
    )


@app.route("/predict", methods=["POST"])
def predict():

    selected_symptoms = request.form.getlist("symptoms")

    disease, confidence, top_predictions = predict_disease(
        selected_symptoms
    )

    return render_template(
        "result.html",
        disease=disease,
        confidence=confidence,
        top_predictions=top_predictions,
        symptoms=selected_symptoms
    )


if __name__ == "__main__":
    app.run(debug=True)