from predict import predict_disease

symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions"
]

result = predict_disease(symptoms)

print("Predicted Disease:", result)