
# Analyse du Projet d'Analyse et de Prédiction du Diabète

## Aperçu de l'application
![Capture d’écran 1](Capture%20d’écran%202025-08-26%20170409.jpg)
![Capture d’écran 2](Capture%20d’écran%202025-08-26%20170436.jpg)
![Capture d’écran 3](Capture%20d’écran%202025-08-26%20170506.jpg)

## 1. Description du Dataset

- **Nom du fichier :** `diabetes_dataset00.csv`
- **Objectif :** Le jeu de données contient des informations médicales et de style de vie anonymisées sur des patients pour prédire l'apparition du diabète.
- **Structure :** Il est composé de nombreuses colonnes (caractéristiques) et d'une colonne cible (`Target`).
- **Caractéristiques notables :**
    - **Données Démographiques :** `Age`, `Ethnicity`, `Socioeconomic Factors`.
    - **Mesures Médicales :** `BMI` (IMC), `Blood Glucose Levels` (Niveau de glucose), `Insulin Levels` (Niveau d'insuline), `Blood Pressure` (Tension artérielle), etc.
    - **Antécédents Médicaux :** `Family History`, `History of PCOS`, `Previous Gestational Diabetes`.
    - **Habitudes de Vie :** `Smoking Status`, `Alcohol Consumption`, `Physical Activity`, `Dietary Habits`.
    - **Marqueurs Spécifiques :** `Genetic Markers`, `Autoantibodies`.
- **Colonne Cible (`Target`) :** C'est la variable que le modèle cherche à prédire. Elle indique le type de diabète ou l'état du patient (par exemple, "Type 1 Diabetes", "Prediabetic", "Secondary Diabetes", etc.).

## 2. Objectif du Projet

L'objectif principal est de construire un système capable de **prédire le risque de diabète** chez un patient en se basant sur ses données médicales, génétiques et de style de vie. Il s'agit d'un problème de **classification**, où le modèle apprend à partir des données existantes pour faire des prédictions sur de nouveaux patients.

Le projet inclut également une interface utilisateur interactive pour faciliter la saisie des données d'un patient et obtenir une prédiction instantanée.

## 3. Actions Réalisées et Fonctionnement

Le projet se déroule en deux phases principales :

**Phase 1 : Entraînement du Modèle (réalisé si aucun modèle n'est déjà sauvegardé)**
1.  **Chargement des données :** Le script charge le dataset `diabetes_dataset00.csv` avec Pandas.
2.  **Prétraitement des données :**
    - Les caractéristiques (`X`) sont séparées de la cible (`y`).
    - Les variables catégorielles (comme "Oui"/"Non" ou "Faible"/"Élevée") sont converties en valeurs numériques pour que le modèle puisse les traiter. C'est ce qu'on appelle l'encodage (ici, `pd.get_dummies`).
    - Les données sont divisées en un ensemble d'entraînement et un ensemble de test.
    - Les caractéristiques numériques sont mises à l'échelle (`StandardScaler`) pour normaliser leur plage de valeurs, ce qui améliore les performances du modèle.
3.  **Entraînement du Modèle :** Un modèle de classification de type **Random Forest** est utilisé. Une recherche sur grille (`GridSearchCV`) est effectuée pour trouver les meilleurs hyperparamètres pour le modèle, optimisant ainsi sa précision.
4.  **Sauvegarde :** Le modèle entraîné (`diabetes_model.joblib`), le scaler (`scaler.joblib`) et la liste des colonnes du modèle (`model_columns.joblib`) sont sauvegardés sur le disque. Cela évite de devoir ré-entraîner le modèle à chaque exécution.

**Phase 2 : Prédiction (Application Interactive)**
1.  **Interface Utilisateur :** Une application web est créée avec Streamlit. Elle offre une interface conviviale dans la barre latérale pour que l'utilisateur puisse entrer les informations d'un patient via des sliders, des menus déroulants, etc.
2.  **Chargement du Modèle :** Au lieu de s'entraîner à nouveau, l'application charge directement le modèle, le scaler et les colonnes sauvegardés.
3.  **Prédiction en Temps Réel :**
    - Les données saisies par l'utilisateur sont formatées pour correspondre exactement à la structure des données d'entraînement.
    - Les données sont mises à l'échelle en utilisant le scaler chargé.
    - Le modèle chargé effectue une prédiction et calcule la probabilité de diabète.
4.  **Affichage et Interprétation :**
    - Le résultat (diabétique ou non) et la probabilité sont affichés à l'utilisateur.
    - Une explication de la prédiction est fournie à l'aide de la bibliothèque **SHAP**, qui montre quels facteurs ont le plus influencé la décision du modèle pour ce patient spécifique.
5.  **Fonctionnalités supplémentaires :** L'application permet de sauvegarder et de charger des profils de patients (fichiers JSON) et de faire des prédictions sur des lots de données via un fichier CSV.

## 4. Technologies Utilisées

- **Langage de programmation :** Python
- **Bibliothèques principales :**
    - **Streamlit :** Pour créer l'application web interactive et l'interface utilisateur.
    - **Pandas :** Pour la manipulation et l'analyse des données (chargement du CSV, traitement des dataframes).
    - **Scikit-learn :** Pour les tâches de Machine Learning (division des données, mise à l'échelle, modèle RandomForest, GridSearchCV).
    - **Joblib :** Pour sauvegarder et charger efficacement le modèle de Machine Learning entraîné.
    - **SHAP :** Pour l'interprétabilité du modèle, c'est-à-dire expliquer les prédictions individuelles.
    - **Matplotlib :** Pour la visualisation des données (graphiques, etc.).

## 5. Comment Exécuter le Projet

1.  **Prérequis :** Assurez-vous que Python est installé sur votre système. Ensuite, installez les bibliothèques nécessaires en ouvrant un terminal ou une invite de commande et en exécutant la commande suivante :
    ```
    **pandas**
    **numpy**
... (the rest of the file is truncated)