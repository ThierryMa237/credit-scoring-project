# 💳 Credit Scoring — Prédiction du Risque de Défaut

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explicabilité-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Présentation

Projet de Machine Learning reproduisant une problématique réelle de **Credit Scoring bancaire**.
L'objectif est de prédire la probabilité qu'un client rembourse ou non son prêt,
à partir de données socio-démographiques, financières et comportementales.

Ce projet simule le travail d'un Data Scientist dans une banque, en respectant
les contraintes métier, réglementaires et techniques du secteur financier.

---

## 🏦 Contexte Métier

Le Credit Scoring est utilisé par les banques pour :
- **Automatiser** les décisions de crédit à grande échelle
- **Réduire les pertes** liées aux défauts de paiement
- **Respecter les réglementations** Bâle II/III et RGPD
- **Expliquer les décisions** aux clients (droit à l'explication)

> Un faux négatif (défaut non détecté) représente une perte financière directe
> pour la banque c'est pourquoi l'AUC-ROC et le Recall sont les métriques
> prioritaires dans ce contexte.

---

## 📊 Dataset

**Source** : [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

| Caractéristique | Valeur |
|---|---|
| Nombre de clients | 307 511 |
| Nombre de variables | 122 |
| Taux de défaut | ~8% |
| Déséquilibre des classes | 92% / 8% |

---

## 🗂️ Architecture du Projet
credit_scoring_project/
│
├── data/
│   ├── raw/                    # Données brutes (non versionnées)
│   └── processed/              # Données preprocessées
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_explainability.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/                     # Modèles sauvegardés
├── images/                     # Visualisations
├── reports/                    # Rapports
├── requirements.txt
└── README.md
---

## 🔬 Méthodologie

### 1. Analyse Exploratoire (EDA)
- Distribution de la variable cible → déséquilibre 92/8
- Détection de l'anomalie `DAYS_EMPLOYED = 365243`
- Analyse des corrélations et des `EXT_SOURCE`
- Taux de défaut par variable catégorielle

### 2. Preprocessing
- Suppression des colonnes avec +50% de valeurs manquantes (41 colonnes)
- Imputation par médiane (variables numériques)
- Imputation par mode (variables catégorielles)
- LabelEncoder (variables binaires)
- One-Hot Encoding (variables multi-modalités)
- StandardScaler (normalisation)

### 3. Feature Engineering
| Feature | Formule | Intérêt Métier |
|---|---|---|
| `DEBT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Niveau d'endettement |
| `ANNUITY_INCOME_RATIO` | AMT_ANNUITY / AMT_INCOME_TOTAL | Effort de remboursement |
| `CREDIT_GOODS_RATIO` | AMT_CREDIT / AMT_GOODS_PRICE | Taux de financement |
| `AGE_YEARS` | DAYS_BIRTH / -365 | Âge en années |
| `YEARS_EMPLOYED` | DAYS_EMPLOYED / -365 | Ancienneté pro |
| `CREDIT_TERM` | AMT_CREDIT / AMT_ANNUITY | Durée du crédit |
| `INCOME_PER_PERSON` | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS | Revenu par personne |
| `EXT_SOURCE_MEAN` | Moyenne EXT_SOURCE 1/2/3 | Score de risque synthétique |

### 4. Modélisation
Cinq modèles comparés :

| Modèle | AUC-ROC |
|---|---|
| Logistic Regression | 0.7456 |
| Decision Tree | 0.5350 |
| Random Forest | 0.7104 |
| Gradient Boosting | 0.7496 |
| **XGBoost (optimisé)** | **0.7563** |

### 5. Explicabilité SHAP
- **Summary plot** : variables les plus influentes globalement
- **Beeswarm plot** : impact par client
- **Waterfall plot** : explication individuelle par client

---

## 📈 Résultats

### Meilleur modèle : XGBoost optimisé

| Métrique | Valeur |
|---|---|
| AUC-ROC | 0.7563 |
| Recall (défauts) | 0.63 |
| Precision (défauts) | 0.18 |
| Accuracy | 0.73 |

### Matrice de confusion
| | Prédit Remboursé | Prédit Défaut |
|---|---|---|
| **Réel Remboursé** | 41 979 ✅ | 14 559 ⚠️ |
| **Réel Défaut** | 1 830 ❌ | 3 135 ✅ |

### Variables les plus importantes (SHAP)
1. `EXT_SOURCE_3`
2. `EXT_SOURCE_2`
3. `EXT_SOURCE_1`
4. `DAYS_BIRTH`
5. `DAYS_EMPLOYED`

---

## 🛠️ Technologies

```python
pandas / numpy          # Manipulation des données
matplotlib / seaborn    # Visualisation
scikit-learn            # Preprocessing et modèles
xgboost                 # Modèle principal
shap                    # Explicabilité
joblib                  # Sauvegarde des modèles
jupyter                 # Environnement de développement
```

---

## 🚀 Comment Reproduire le Projet

```bash
# 1. Clone le dépôt
git clone https://github.com/ThierryMa237/credit-scoring-project.git
cd credit-scoring-project

# 2. Crée l'environnement virtuel
python3 -m venv credit_scoring_env
source credit_scoring_env/bin/activate

# 3. Installe les dépendances
pip install -r requirements.txt

# 4. Télécharge le dataset
kaggle competitions download -c home-credit-default-risk
mv home-credit-default-risk.zip data/raw/
cd data/raw && unzip home-credit-default-risk.zip

# 5. Lance les notebooks dans l'ordre
jupyter notebook
```

---

## 🔭 Pistes d'Amélioration

- **LightGBM / CatBoost** : alternatives potentiellement plus performantes
- **Optimisation Bayésienne** : plus efficace que RandomizedSearchCV
- **Calibration des probabilités** : Platt Scaling ou Isotonic Regression
- **SMOTE** : sur-échantillonnage pour mieux gérer le déséquilibre
- **Fichiers supplémentaires Kaggle** : bureau.csv, previous_application.csv
- **Détection de dérive** : monitoring du modèle en production
- **Déploiement FastAPI** : API REST pour les prédictions en temps réel
- **Dashboard Streamlit** : interface interactive pour les équipes métier

---

👤 **Thierry Manuel Zibi Ondobo**
Étudiant Ingénieur Data Science, Junia-ISEN

**N.B** Ce projet a uniquement été réalisé pour monter en compétences et tenir un profil solide pour la recherche de mon stage et aussi l'amélioration de mon profil de data scientist

[![GitHub](https://img.shields.io/badge/GitHub-ThierryMa237-black)](https://github.com/ThierryMa237)
