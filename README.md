
# Analyse du Projet d'Analyse et de Prédiction du Diabète

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
    **scikit-learn**
    **streamlit**
    **matplotlib**
    **shap**
    **seaborn**
    pip install streamlit pandas scikit-learn joblib shap matplotlib
    ```

2.  **Lancement de l'application :**
    - Placez-vous dans le répertoire du projet (`C:\Users\HP\Pictures\fouille_de_données`) via le terminal.
    - Exécutez la commande suivante :
    ```
    streamlit run diabetes_analysis.py
    ```

3.  **Utilisation :**
    - Une page web devrait s'ouvrir automatiquement dans votre navigateur.
    - Utilisez la barre latérale pour entrer les données d'un patient.
    - Cliquez sur le bouton "Lancer le diagnostic" pour voir la prédiction.
    - Vous pouvez également naviguer vers le "Tableau de Bord" pour voir des informations sur l'importance des caractéristiques du modèle.
Résumé des Modifications Apportées au Fichier `diabetes_analysis.py`

Ce document détaille toutes les modifications apportées au script `diabetes_analysis.py` depuis le début de notre interaction, incluant les fonctionnalités ajoutées, les corrections de bugs, et les ajustements de l'interface utilisateur.

---

**1. Traduction Initiale des Noms de Colonnes (Tentative et Correction)**

*   **Objectif :** Faciliter l'utilisation du script pour les utilisateurs francophones en traduisant les noms de colonnes et certaines valeurs.
*   **Action :**
    *   Initialement, une tentative a été faite pour traduire des noms de colonnes comme 'Pregnancies', 'Glucose', etc., directement dans le code.
    *   **Problème rencontré :** Une erreur de remplacement (`replace` tool failed) est survenue car la chaîne de caractères à remplacer ne correspondait pas exactement au contenu du fichier.
    *   **Correction :** Le fichier a été relu pour obtenir son contenu exact, puis des remplacements ciblés ont été effectués.
        *   Les noms de colonnes dans la fonction `user_input_features` (par exemple, 'Family History', 'Smoking Status', 'BMI', 'Insulin Levels', 'Blood Pressure', 'Waist Circumference') ont été traduits en français (par exemple, 'AntecedentsFamiliaux', 'StatutFumeur', 'IMC', 'NiveauxInsuline', 'PressionArterielle', 'TourTaille') pour les clés du dictionnaire `data` et les étiquettes de l'interface utilisateur.
        *   La colonne 'Target' a été traduite en 'Cible' dans les sections d'entraînement et de visualisation des données.

---

**4. Extension des Champs de Saisie Utilisateur**

*   **Objectif :** Permettre à l'utilisateur de saisir des données pour toutes les colonnes du dataset `diabetes_dataset00.csv` via l'interface Streamlit.
*   **Action :**
    *   Toutes les colonnes du fichier CSV (à l'exception de la colonne cible) ont été identifiées.
    *   La fonction `user_input_features` a été étendue pour inclure des widgets de saisie (sliders pour les valeurs numériques, selectboxes pour les catégories) pour chaque colonne.
    *   Les étiquettes de l'interface utilisateur pour ces nouveaux champs ont été traduites en français.
    *   Des mappages ont été créés pour convertir les entrées de chaînes de caractères des selectboxes en valeurs numériques (par exemple, "Non"/"Oui" en 0/1).
    *   **Problème rencontré :** Une erreur de remplacement (`replace` tool failed) est survenue lors de l'application de cette modification, nécessitant une relecture du fichier pour un remplacement exact.

---
**6. Vérification des Performances du Modèle (Tentative et Annulation)**

*   **Objectif :** Afficher la performance du modèle (précision) sur l'interface utilisateur.
*   **Action (tentée) :**
    *   Importation de `accuracy_score` de `sklearn.metrics`.
    *   Modification de la fonction `train_and_save_model` pour sauvegarder `X_test_scaled` et `y_test` en tant que fichiers `.joblib`.
    *   Modification du bloc de chargement/entraînement du modèle pour charger `X_test_scaled` et `y_test`.
    *   Ajout d'une section "Performance du Modèle" dans le "Tableau de Bord" pour calculer et afficher la précision.
    *   **Problème rencontré :** `FileNotFoundError` car `X_test_scaled.joblib` et `y_test.joblib` n'existaient pas lors du premier chargement (nécessitant une suppression manuelle des fichiers de modèle pour forcer le réentraînement).
    *   **Problème rencontré :** `NameError` car `X_test_scaled` n'était pas défini dans `train_and_save_model` avant d'être sauvegardé.
    *   **Correction (pour `NameError`) :** Ajout de la ligne `X_test_scaled = scaler.transform(X_test)` dans `train_and_save_model`.
    *   **Annulation :** L'utilisateur a demandé d'annuler toutes les modifications liées à la vérification des performances. Toutes les modifications de cette phase ont été annulées, y compris les importations, les sauvegardes/chargements de fichiers `.joblib` supplémentaires, et la section d'affichage des performances.

---

**7. Traduction des Colonnes du Tableau "Données du patient saisies"**

*   **Objectif :** Traduire les noms de colonnes du tableau affiché "Données du patient saisies" sur l'interface utilisateur.
*   **Action :**
    *   Une nouvelle section a été ajoutée juste avant l'affichage du `input_df`.
    *   Un dictionnaire `column_name_mapping` a été créé, mappant les noms de colonnes anglais (utilisés comme clés dans le DataFrame) à leurs équivalents français (utilisés comme étiquettes dans l'interface utilisateur).
    *   Le DataFrame `input_df` est renommé en `input_df_display` en utilisant ce mappage avant d'être affiché avec `st.write()`.

---

**Instructions d'Exécution de l'Application**

Pour exécuter cette application Streamlit, suivez les étapes ci-dessous :

**1. Logiciels Prérequis :**
*   **Python :** Assurez-vous d'avoir Python 3.7 ou une version ultérieure installée sur votre système. Vous pouvez le télécharger depuis le site officiel de Python (python.org).

**2. Installation des Bibliothèques Python :**
Ouvrez votre terminal ou invite de commande, naviguez jusqu'au répertoire où se trouve le fichier `diabetes_analysis.py` et exécutez la commande suivante pour installer toutes les bibliothèques nécessaires :

```bash
pip install pandas scikit-learn streamlit joblib matplotlib shap
```

*   **`pandas` :** Utilisé pour la manipulation et l'analyse des données (lecture du CSV, création de DataFrames).
*   **`scikit-learn` (sklearn) :** Bibliothèque d'apprentissage automatique pour la modélisation (séparation des données, mise à l'échelle, entraînement du modèle RandomForestClassifier, GridSearchCV pour l'optimisation des hyperparamètres).
*   **`streamlit` :** Framework pour la création d'applications web interactives (l'interface utilisateur que vous voyez).
*   **`joblib` :** Utilisé pour sauvegarder et charger les modèles entraînés et les scalers, ainsi que les colonnes du modèle.
*   **`matplotlib` :** Bibliothèque de visualisation de données, utilisée notamment pour les graphiques SHAP et les distributions de données.
*   **`shap` :** Utilisé pour l'interprétabilité du modèle, notamment pour expliquer les prédictions individuelles via les graphiques SHAP (SHapley Additive exPlanations).

**3. Exécution de l'Application :**
Une fois toutes les bibliothèques installées, exécutez l'application en utilisant la commande suivante dans le même terminal :

```bash
streamlit run diabetes_analysis.py
```

Cette commande ouvrira l'application dans votre navigateur web par défaut.

**4. Utilisation de l'Application :**
*   **Section "Prédiction" :** Utilisez les champs de la barre latérale pour saisir les données du patient. Cliquez sur "Lancer le diagnostic" pour obtenir une prédiction et une explication SHAP.
*   **Section "Tableau de Bord" :** Visualisez l'importance des caractéristiques du modèle et la distribution des données d'entraînement.

---