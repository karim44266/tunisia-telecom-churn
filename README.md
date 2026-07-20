# Tunisia Telecom Churn Prediction

Projet de machine learning pour prédire le désabonnement (churn) des clients télécom, avec une application interactive Streamlit permettant de tester le modèle en temps réel.

## 📊 Contexte

La prédiction de churn est un cas d'usage classique en data science business : identifier à l'avance les clients susceptibles de quitter un opérateur télécom, pour permettre des actions de rétention ciblées.

Ce projet couvre le pipeline complet : exploration des données, préparation, modélisation, évaluation, explicabilité (SHAP) et déploiement via une app web.

## 🗂️ Structure du projet

```
tunisia-telecom-churn/
├── notebooks/
│   └── 01_eda.ipynb        # Exploration, préparation, modélisation, évaluation
├── src/
│   └── churn_model.pkl     # Modèle entraîné sauvegardé
├── app.py                  # Application Streamlit
├── requirements.txt
└── README.md
```

## 🔍 Approche

1. **Nettoyage des données** : conversion de `TotalCharges`, gestion des valeurs manquantes (cohérentes avec `tenure == 0`)
2. **Analyse exploratoire (EDA)** : distribution du churn, corrélations, comparaisons par contrat/ancienneté/facturation
3. **Feature engineering** : encodage one-hot des variables catégorielles
4. **Modélisation** : Logistic Regression (avec Pipeline scaler + modèle) et XGBoost
5. **Évaluation** : accuracy, ROC-AUC, recall — avec un focus particulier sur le recall des churners plutôt que l'accuracy brute
6. **Explicabilité** : SHAP pour comprendre les facteurs qui poussent un client vers le churn
7. **Déploiement** : application Streamlit pour tester le modèle sur un profil client donné

## 📈 Résultats

| Modèle | Accuracy | ROC-AUC | Recall (churn) |
|---|---|---|---|
| Logistic Regression | ~74.0% | ~0.84 | — |
| XGBoost | ~75.7% | ~0.84 | ~78% |

**Note méthodologique importante** : sur ce jeu de données, une baseline naïve qui prédit toujours "pas de churn" atteint déjà ~73.5% d'accuracy sans jamais détecter un seul churner. L'accuracy brute est donc un indicateur trompeur ici. Un test de recherche d'hyperparamètres purement optimisé pour l'accuracy a atteint ~79.8%, mais au prix d'un recall churn chuté à 51% — c'est-à-dire qu'il manquait la moitié des vrais churners.

**Choix retenu** : privilégier le recall churners (XGBoost, 78% de recall) plutôt que l'accuracy maximale, car dans un contexte business, manquer un client à risque coûte plus cher qu'une fausse alerte. Ces résultats sont cohérents avec les benchmarks publiés sur ce jeu de données (Telco Customer Churn), qui plafonnent généralement autour de 80-82% d'accuracy.

## ⚠️ Limites connues

- Jeu de données déséquilibré (classe "churn" minoritaire)
- Pas de feature engineering avancé (pas d'interactions, pas de features temporelles complexes)
- Modèle non recalibré (les scores de probabilité peuvent être surconfiants sur les cas à haut risque)

## 🚀 Installation

```bash
git clone <url-du-repo>
cd tunisia-telecom-churn
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## ▶️ Utilisation

**Notebook (exploration et entraînement)**
```bash
jupyter notebook
# Ouvrir notebooks/01_eda.ipynb, puis Run → Run All Cells
```

**Application Streamlit**
```bash
streamlit run app.py
```

## 🛠️ Stack technique

Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib/Seaborn, Joblib

## 📌 Pistes d'amélioration futures

- Calibration des probabilités (`CalibratedClassifierCV`)
- Feature engineering additionnel (segments clients, interactions)
- Validation croisée plus poussée
- Suivi de la dérive du modèle en production (monitoring)
