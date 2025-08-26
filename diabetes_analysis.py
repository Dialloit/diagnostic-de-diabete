import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import shap
import json

# --- CONFIGURATION & MODÈLE DE MACHINE LEARNING ---

st.set_page_config(page_title="Prédiction du Diabète", layout="wide")

COLUMN_NAME_MAPPING = {
    'Family History': 'Antécédents familiaux',
    'Smoking Status': 'Statut de fumeur',
    'Previous Gestational Diabetes': 'Diabète gestationnel antérieur',
    'History of PCOS': 'Antécédents de SOPK',
    'Genetic Markers': 'Marqueurs génétiques',
    'Autoantibodies': 'Auto-anticorps',
    'Environmental Factors': 'Facteurs environnementaux',
    'Early Onset Symptoms': 'Symptômes précoces',
    'Age': 'Âge',
    'BMI': 'IMC',
    'Blood Glucose Levels': 'Niveau de glucose sanguin',
    'Insulin Levels': 'Niveaux d\'insuline',
    'Blood Pressure': 'Pression artérielle (systolique)',
    'Waist Circumference': 'Tour de taille (cm)',
    'Cholesterol Levels': 'Niveaux de cholestérol',
    'Physical Activity': 'Activité physique',
    'Dietary Habits': 'Habitudes alimentaires',
    'Alcohol Consumption': 'Consommation d\'alcool',
    'Ethnicity': 'Ethnicité',
    'Socioeconomic Factors': 'Facteurs socio-économiques',
    'Pregnancy History': 'Antécédents de grossesse',
    'Weight Gain During Pregnancy': 'Prise de poids pendant la grossesse',
    'Pancreatic Health': 'Santé pancréatique',
    'Pulmonary Function': 'Fonction pulmonaire',
    'Cystic Fibrosis Diagnosis': 'Diagnostic de mucoviscidose',
    'Steroid Use History': 'Antécédents d\'utilisation de stéroïdes',
    'Genetic Testing': 'Tests génétiques',
    'Neurological Assessments': 'Évaluations neurologiques',
    'Liver Function Tests': 'Tests de fonction hépatique',
    'Digestive Enzyme Levels': 'Niveaux d\'enzymes digestives',
    'Urine Test': 'Test urinaire',
    'Glucose Tolerance Test': 'Test de tolérance au glucose',
    'Birth Weight': 'Poids à la naissance (g)',
    'Glucose_Insulin_Ratio': 'Ratio Glucose/Insuline',
    'Age_BMI_Interaction': 'Interaction Âge*IMC'
}

@st.cache_data
def train_and_save_model():
    st.info("Le modèle n'a pas été trouvé. Démarrage de l'entraînement... (cela peut prendre quelques minutes)")
    df = pd.read_csv('diabetes_dataset00.csv')
    X = df.drop('Target', axis=1)
    y = df['Target']

    median_insulin = X['Insulin Levels'][X['Insulin Levels'] > 0].median()
    joblib.dump(median_insulin, 'median_insulin.joblib')
    X['Insulin Levels'] = X['Insulin Levels'].replace(0, median_insulin)
    X['Glucose_Insulin_Ratio'] = X['Blood Glucose Levels'] / X['Insulin Levels']
    X['Age_BMI_Interaction'] = X['Age'] * X['BMI']

    X_dummies = pd.get_dummies(X, drop_first=True)
    model_columns = X_dummies.columns
    joblib.dump(model_columns, 'model_columns.joblib')
    
    X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.2, random_state=42, stratify=y)
    joblib.dump((X_test, y_test), 'test_data.joblib')

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_rf_model = grid_search.best_estimator_
    joblib.dump(best_rf_model, 'diabetes_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    st.success("Entraînement terminé et modèle sauvegardé !")
    return best_rf_model, scaler, model_columns, median_insulin, (X_test, y_test)

# Charger ou entraîner le modèle
if not os.path.exists('diabetes_model.joblib'):
    model, scaler, model_columns, median_insulin, test_data = train_and_save_model()
else:
    model = joblib.load('diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
    if os.path.exists('median_insulin.joblib'):
        median_insulin = joblib.load('median_insulin.joblib')
    else:
        df_temp = pd.read_csv('diabetes_dataset00.csv')
        median_insulin = df_temp['Insulin Levels'][df_temp['Insulin Levels'] > 0].median()
    if os.path.exists('test_data.joblib'):
        test_data = joblib.load('test_data.joblib')
    else:
        st.warning("Données de test non trouvées. Le tableau de bord sera limité.")
        test_data = None

# --- INTERFACE UTILISATEUR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", ["🔮 Prédiction", "📊 Tableau de Bord"])

if page == "🔮 Prédiction":
    st.sidebar.header('Saisir les données du patient')
    
    def user_input_features():
        if 'Family History' not in st.session_state: st.session_state['Family History'] = "Non"
        if 'Smoking Status' not in st.session_state: st.session_state['Smoking Status'] = "Non"
        if 'Previous Gestational Diabetes' not in st.session_state: st.session_state['Previous Gestational Diabetes'] = "Non"
        if 'History of PCOS' not in st.session_state: st.session_state['History of PCOS'] = "Non"
        if 'Genetic Markers' not in st.session_state: st.session_state['Genetic Markers'] = "Non"
        if 'Autoantibodies' not in st.session_state: st.session_state['Autoantibodies'] = "Non"
        if 'Environmental Factors' not in st.session_state: st.session_state['Environmental Factors'] = "Absent"
        if 'Early Onset Symptoms' not in st.session_state: st.session_state['Early Onset Symptoms'] = "Non"
        if 'Age' not in st.session_state: st.session_state['Age'] = 30
        if 'BMI' not in st.session_state: st.session_state['BMI'] = 25.0
        if 'Blood Glucose Levels' not in st.session_state: st.session_state['Blood Glucose Levels'] = 100
        if 'Insulin Levels' not in st.session_state: st.session_state['Insulin Levels'] = 80
        if 'Blood Pressure' not in st.session_state: st.session_state['Blood Pressure'] = 120
        if 'Waist Circumference' not in st.session_state: st.session_state['Waist Circumference'] = 90
        if 'Cholesterol Levels' not in st.session_state: st.session_state['Cholesterol Levels'] = 150
        if 'Physical Activity' not in st.session_state: st.session_state['Physical Activity'] = "Modérée"
        if 'Dietary Habits' not in st.session_state: st.session_state['Dietary Habits'] = "Saines"
        if 'Alcohol Consumption' not in st.session_state: st.session_state['Alcohol Consumption'] = "Modérée"
        if 'Ethnicity' not in st.session_state: st.session_state['Ethnicity'] = "Low Risk"
        if 'Socioeconomic Factors' not in st.session_state: st.session_state['Socioeconomic Factors'] = "Medium"
        if 'Pregnancy History' not in st.session_state: st.session_state['Pregnancy History'] = 0
        if 'Weight Gain During Pregnancy' not in st.session_state: st.session_state['Weight Gain During Pregnancy'] = 15
        if 'Pancreatic Health' not in st.session_state: st.session_state['Pancreatic Health'] = 70
        if 'Pulmonary Function' not in st.session_state: st.session_state['Pulmonary Function'] = 75
        if 'Cystic Fibrosis Diagnosis' not in st.session_state: st.session_state['Cystic Fibrosis Diagnosis'] = "Non"
        if 'Steroid Use History' not in st.session_state: st.session_state['Steroid Use History'] = "Non"
        if 'Genetic Testing' not in st.session_state: st.session_state['Genetic Testing'] = "Négatif"
        if 'Neurological Assessments' not in st.session_state: st.session_state['Neurological Assessments'] = "Normal"
        if 'Liver Function Tests' not in st.session_state: st.session_state['Liver Function Tests'] = "Normal"
        if 'Digestive Enzyme Levels' not in st.session_state: st.session_state['Digestive Enzyme Levels'] = "Normal"
        if 'Urine Test' not in st.session_state: st.session_state['Urine Test'] = "Normal"
        if 'Glucose Tolerance Test' not in st.session_state: st.session_state['Glucose Tolerance Test'] = "Normal"
        if 'Birth Weight' not in st.session_state: st.session_state['Birth Weight'] = 3000

        data = {}
        binary_map = {"Non": 0, "Oui": 1}
        present_absent_map = {"Absent": 0, "Présent": 1}
        low_mod_high_map = {"Faible": 0, "Modérée": 1, "Élevée": 2}
        healthy_unhealthy_map = {"Saines": 0, "Malsaines": 1}
        normal_abnormal_map = {"Normal": 0, "Anormal": 1}
        neg_pos_map = {"Négatif": 0, "Positif": 1}
        urine_test_map = {"Normal": 0, "Cétones présentes": 1, "Glucose présent": 2, "Protéines présentes": 3}
        ethnicity_map = {"Low Risk": 0, "High Risk": 1}
        socioeconomic_map = {"Low": 0, "Medium": 1, "High": 2}

        st.sidebar.subheader("Facteurs de Risque Principaux")
        data['Family History'] = binary_map[st.sidebar.selectbox('Antécédents familiaux', ["Non", "Oui"], key='Family History')]
        data['Smoking Status'] = binary_map[st.sidebar.selectbox('Statut de fumeur', ["Non", "Oui"], key='Smoking Status')]
        data['Previous Gestational Diabetes'] = binary_map[st.sidebar.selectbox('Diabète gestationnel antérieur', ["Non", "Oui"], key='Previous Gestational Diabetes')]
        data['History of PCOS'] = binary_map[st.sidebar.selectbox('Antécédents de SOPK', ["Non", "Oui"], key='History of PCOS')]
        data['Genetic Markers'] = binary_map[st.sidebar.selectbox('Marqueurs génétiques', ["Non", "Oui"], key='Genetic Markers')]
        data['Autoantibodies'] = binary_map[st.sidebar.selectbox('Auto-anticorps', ["Non", "Oui"], key='Autoantibodies')]
        data['Environmental Factors'] = present_absent_map[st.sidebar.selectbox('Facteurs environnementaux', ["Absent", "Présent"], key='Environmental Factors')]
        data['Early Onset Symptoms'] = binary_map[st.sidebar.selectbox('Symptômes précoces', ["Non", "Oui"], key='Early Onset Symptoms')]

        st.sidebar.subheader("Mesures Biométriques")
        data['Age'] = st.sidebar.slider('Âge', 1, 120, key='Age')
        data['BMI'] = st.sidebar.slider('IMC', 0.0, 70.0, 0.1, key='BMI')
        data['Blood Glucose Levels'] = st.sidebar.slider('Niveau de glucose sanguin', 0, 300, key='Blood Glucose Levels')
        data['Insulin Levels'] = st.sidebar.slider('Niveaux d\'insuline', 0, 900, key='Insulin Levels')
        data['Blood Pressure'] = st.sidebar.slider('Pression artérielle (systolique)', 0, 200, key='Blood Pressure')
        data['Waist Circumference'] = st.sidebar.slider('Tour de taille (cm)', 0, 200, key='Waist Circumference')
        data['Cholesterol Levels'] = st.sidebar.slider('Niveaux de cholestérol', 0, 300, key='Cholesterol Levels')

        st.sidebar.subheader("Habitudes de Vie et Facteurs Sociaux")
        data['Physical Activity'] = low_mod_high_map[st.sidebar.selectbox('Activité physique', ["Faible", "Modérée", "Élevée"], key='Physical Activity')]
        data['Dietary Habits'] = healthy_unhealthy_map[st.sidebar.selectbox('Habitudes alimentaires', ["Saines", "Malsaines"], key='Dietary Habits')]
        data['Alcohol Consumption'] = low_mod_high_map[st.sidebar.selectbox('Consommation d\'alcool', ["Faible", "Modérée", "Élevée"], key='Alcohol Consumption')]
        data['Ethnicity'] = ethnicity_map[st.sidebar.selectbox('Ethnicité', ["Low Risk", "High Risk"], key='Ethnicity')]
        data['Socioeconomic Factors'] = socioeconomic_map[st.sidebar.selectbox('Facteurs socio-économiques', ["Low", "Medium", "High"], key='Socioeconomic Factors')]

        st.sidebar.subheader("Historique Médical et Tests Spécifiques")
        data['Pregnancy History'] = st.sidebar.slider('Antécédents de grossesse', 0, 10, key='Pregnancy History')
        data['Weight Gain During Pregnancy'] = st.sidebar.slider('Prise de poids pendant la grossesse', 0, 50, key='Weight Gain During Pregnancy')
        data['Pancreatic Health'] = st.sidebar.slider('Santé pancréatique', 0, 100, key='Pancreatic Health')
        data['Pulmonary Function'] = st.sidebar.slider('Fonction pulmonaire', 0, 100, key='Pulmonary Function')
        data['Cystic Fibrosis Diagnosis'] = binary_map[st.sidebar.selectbox('Diagnostic de mucoviscidose', ["Non", "Oui"], key='Cystic Fibrosis Diagnosis')]
        data['Steroid Use History'] = binary_map[st.sidebar.selectbox('Antécédents d\'utilisation de stéroïdes', ["Non", "Oui"], key='Steroid Use History')]
        data['Genetic Testing'] = neg_pos_map[st.sidebar.selectbox('Tests génétiques', ["Négatif", "Positif"], key='Genetic Testing')]
        data['Neurological Assessments'] = normal_abnormal_map[st.sidebar.selectbox('Évaluations neurologiques', ["Normal", "Anormal"], key='Neurological Assessments')]
        data['Liver Function Tests'] = normal_abnormal_map[st.sidebar.selectbox('Tests de fonction hépatique', ["Normal", "Anormal"], key='Liver Function Tests')]
        data['Digestive Enzyme Levels'] = normal_abnormal_map[st.sidebar.selectbox('Niveaux d\'enzymes digestives', ["Normal", "Anormal"], key='Digestive Enzyme Levels')]
        data['Urine Test'] = urine_test_map[st.sidebar.selectbox('Test urinaire', ["Normal", "Cétones présentes", "Glucose présent", "Protéines présentes"], key='Urine Test')]
        data['Glucose Tolerance Test'] = normal_abnormal_map[st.sidebar.selectbox('Test de tolérance au glucose', ["Normal", "Anormal"], key='Glucose Tolerance Test')]
        data['Birth Weight'] = st.sidebar.slider('Poids à la naissance (g)', 0, 5000, key='Birth Weight')
        
        insulin_level = data['Insulin Levels'] if data['Insulin Levels'] > 0 else median_insulin
        data['Glucose_Insulin_Ratio'] = data['Blood Glucose Levels'] / insulin_level if insulin_level > 0 else 0
        data['Age_BMI_Interaction'] = data['Age'] * data['BMI']

        return data

    st.title('Prédiction de Diagnostic du Diabète')
    user_input_data = user_input_features()

    st.sidebar.subheader("Gestion des Profils")
    
    if st.sidebar.button("Sauvegarder Profil"):
        profile_data = {key: st.session_state[key] for key in st.session_state if key not in ["FormSubmitter:Lancer le diagnostic", "FormSubmitter:Sauvegarder Profil", "FormSubmitter:Charger Profil"]}
        file_name = st.sidebar.text_input("Nom du fichier de sauvegarde", "patient_profile.json")
        if file_name:
            with open(file_name, "w") as f:
                json.dump(profile_data, f)
            st.sidebar.success(f"Profil sauvegardé sous {file_name}")
        else:
            st.sidebar.warning("Veuillez entrer un nom de fichier.")

    uploaded_file = st.sidebar.file_uploader("Charger Profil", type=["json"])
    if uploaded_file is not None:
        try:
            profile_data = json.load(uploaded_file)
            for key, value in profile_data.items():
                if key in st.session_state:
                    st.session_state[key] = value
            st.sidebar.success("Profil chargé avec succès !")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement du profil : {e}")

    st.subheader('Données du patient saisies (incluant les nouvelles caractéristiques) :')
    
    display_data = user_input_data.copy()
    input_df_display = pd.DataFrame([display_data]).rename(columns=COLUMN_NAME_MAPPING)
    st.write(input_df_display)

    if st.button('Lancer le diagnostic'):
        errors = []
        if not (1 <= user_input_data['Age'] <= 120):
            errors.append("L'âge doit être compris entre 1 et 120.")
        if not (0.0 <= user_input_data['BMI'] <= 70.0):
            errors.append("L'IMC doit être compris entre 0.0 et 70.0.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            input_df = pd.DataFrame([user_input_data])
            input_df_dummies = pd.get_dummies(input_df)
            input_df_processed = input_df_dummies.reindex(columns=model_columns, fill_value=0)
            
            input_scaled = scaler.transform(input_df_processed)
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            st.subheader('Résultat du Diagnostic')
            if prediction[0] == 1:
                st.error(f'Le patient est susceptible d\'être diabétique (Probabilité : {prediction_proba[0][1]*100:.2f}%)')
            else:
                st.success(f'Le patient n\'est probablement pas diabétique (Probabilité : {prediction_proba[0][0]*100:.2f}%)')
            
            try:
                st.subheader("Explication de la Prédiction")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                st.write("Ce graphique montre les facteurs qui ont poussé la prédiction vers un résultat positif (en rouge) ou négatif (en bleu).")
                
                if isinstance(shap_values, list):
                    # Cas où l'explainer renvoie les valeurs pour chaque classe
                    shap_values_to_plot = shap_values[1][0]  # Première observation, classe 1
                    expected_value_to_plot = explainer.expected_value[1]
                else:
                    # Cas où l'explainer ne renvoie qu'une seule valeur
                    shap_values_to_plot = shap_values[0]  # Première observation
                    expected_value_to_plot = explainer.expected_value
                
                # Création du force plot avec la nouvelle API
                force_plot = shap.force_plot(
                    base_value=expected_value_to_plot,
                    shap_values=shap_values_to_plot,
                    features=input_df_processed.iloc[0],
                    feature_names=input_df_processed.columns.tolist(),
                    matplotlib=True,
                    show=False
                )
                
                st.pyplot(force_plot)
                plt.close()
            except Exception as e:
                st.warning(f"Impossible de générer le graphique d'explication SHAP. Erreur : {e}")

    st.subheader("Prédiction par Lot")
    uploaded_file_batch = st.file_uploader("Télécharger un fichier CSV pour la prédiction par lot", type=["csv"])

    if uploaded_file_batch is not None:
        try:
            batch_df = pd.read_csv(uploaded_file_batch)
            
            batch_df['Insulin Levels'] = batch_df['Insulin Levels'].replace(0, median_insulin)
            batch_df['Glucose_Insulin_Ratio'] = batch_df['Blood Glucose Levels'] / batch_df['Insulin Levels']
            batch_df['Age_BMI_Interaction'] = batch_df['Age'] * batch_df['BMI']

            batch_df_dummies = pd.get_dummies(batch_df)
            batch_df_processed = batch_df_dummies.reindex(columns=model_columns, fill_value=0)
            batch_df_processed = batch_df_processed[model_columns] 

            batch_scaled = scaler.transform(batch_df_processed)
            predictions = model.predict(batch_scaled)
            predictions_proba = model.predict_proba(batch_scaled)

            results = []
            for i, prediction in enumerate(predictions):
                result_text = "Diabétique" if prediction == 1 else "Non Diabétique"
                probability = predictions_proba[i][1] if prediction == 1 else predictions_proba[i][0]
                results.append({
                    "Patient Index": batch_df.index[i],
                    "Prédiction": result_text,
                    "Probabilité": f"{probability*100:.2f}%"
                })
            
            results_df = pd.DataFrame(results)
            st.subheader("Résultats de la Prédiction par Lot")
            st.dataframe(results_df)

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier CSV : {e}")

elif page == "📊 Tableau de Bord":
    st.title("Tableau de Bord - Analyse de Performance du Modèle")

    st.subheader("Importance des Caractéristiques")
    if len(model.feature_importances_) == len(model_columns):
        feature_importances = pd.DataFrame(model.feature_importances_, index=model_columns, columns=['Importance']).sort_values('Importance', ascending=False)
        st.write("Ce graphique montre les facteurs qui influencent le plus les décisions du modèle.")
        st.bar_chart(feature_importances.head(20))
    else:
        st.warning("Impossible d'afficher l'importance des caractéristiques.")

    if test_data:
        X_test, y_test = test_data
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)

        st.subheader("Matrice de Confusion")
        st.write("La matrice de confusion montre les prédictions correctes et incorrectes pour chaque classe.")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm, cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        ax_cm.set_xlabel('Prédiction')
        ax_cm.set_ylabel('Vraie valeur')
        st.pyplot(fig_cm)

        st.subheader("Courbe ROC")
        st.write("La courbe ROC évalue la capacité du modèle à distinguer les classes. Un modèle est d'autant meilleur que la courbe est proche du coin supérieur gauche.")
        fig_roc, ax_roc = plt.subplots()
        
        y_test_bin = label_binarize(y_test, classes=model.classes_)
        n_classes = y_test_bin.shape[1]

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f'Classe {model.classes_[i]} (AUC = {roc_auc:.2f})')
        
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('Taux de Faux Positifs')
        ax_roc.set_ylabel('Taux de Vrais Positifs')
        ax_roc.set_title('Courbe ROC Multi-classe')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.subheader("Distribution des Données d'Entraînement")
    df_original = pd.read_csv('diabetes_dataset00.csv')
    
    df_original['Insulin Levels'] = df_original['Insulin Levels'].replace(0, median_insulin)
    df_original['Glucose_Insulin_Ratio'] = df_original['Blood Glucose Levels'] / df_original['Insulin Levels']
    df_original['Age_BMI_Interaction'] = df_original['Age'] * df_original['BMI']
    
    features_list = df_original.columns.drop('Target')

    feature_to_show = st.selectbox(
        "Choisissez une caractéristique à visualiser", 
        features_list,
        format_func=lambda x: COLUMN_NAME_MAPPING.get(x, x)
    )
    
    fig_dist, ax_dist = plt.subplots()
    ax_dist.hist(df_original[feature_to_show], bins=20, edgecolor='black')
    ax_dist.set_title(f'Distribution de {COLUMN_NAME_MAPPING.get(feature_to_show, feature_to_show)}')
    ax_dist.set_xlabel("Valeur")
    ax_dist.set_ylabel("Fréquence")
    st.pyplot(fig_dist)
