import pandas as pd
import numpy as np
import joblib

# Charger le modèle entraîné et le scaler
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')
model_columns = joblib.load('model_columns.joblib')

def predict_diabetes(data):
    # Convertir les données d'entrée en un DataFrame pandas
    df = pd.DataFrame([data], columns=model_columns)

    # Mettre à l'échelle les caractéristiques numériques
    numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Faire la prédiction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    # Interpréter la prédiction
    if prediction[0] == 1:
        result = "Diabétique"
        probability = prediction_proba[0][1]
    else:
        result = "Non-Diabétique"
        probability = prediction_proba[0][0]

    return result, probability

if __name__ == "__main__":
    # Exemple d'utilisation :
    # Remplacer par les données réelles du patient
    patient_data = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }

    prediction_result, probability_score = predict_diabetes(patient_data)
    print(f"Prédiction : {prediction_result}")
    print(f"Probabilité : {probability_score:.2f}")