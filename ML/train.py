"""
train.py
Entraîne UNIQUEMENT le modèle final retenu : Random Forest + class_weight
"balanced". Choisi après comparaison dans training_models.py (meilleure
precision avec très peu de fausses alertes, robuste au déséquilibre de
classes du dataset AI4I 2020).
Ce script est autonome : il repart des données brutes, refait le feature
engineering, entraîne, évalue brièvement, et sauvegarde tout ce qui est
nécessaire pour predict.py.
"""
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)

# -----------------------------------------------------------------
# 0. Configuration des chemins
# -----------------------------------------------------------------
BASE_DIR = r"E:\AKWPJ\EEEIA_2026\Projet_Agent_IA"
DATA_PATH = BASE_DIR + r"\ai4i2020.csv"
MODEL_DIR = BASE_DIR + r"\models\\"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 1. Chargement + feature engineering (identique à training_models.py,
#    pour que ce script reste indépendant et reproductible seul)
# -----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

type_map = {"L": 0, "M": 1, "H": 2}
df["Type_encoded"] = df["Type"].map(type_map)
df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (np.pi / 30)
df["strain"] = df["Torque [Nm]"] * df["Tool wear [min]"]

feature_cols = [
    "Type_encoded",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "temp_diff", "power_W", "strain",
]
target_col = "Machine failure"

X = df[feature_cols]
y = df[target_col]

# -----------------------------------------------------------------
# 2. Split train/test stratifié (mêmes paramètres que training_models.py,
#    pour rester comparable)
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=8000, test_size=2000, stratify=y, random_state=42
)

# -----------------------------------------------------------------
# 3. Entraînement du Random Forest final
#    Pas de scaling nécessaire : les arbres de décision travaillent par
#    seuils sur chaque variable, pas par distance -> insensibles à l'échelle.
#    class_weight="balanced" : pénalise davantage les erreurs sur la classe
#    minoritaire (pannes, 3.39% des cas) pendant l'entraînement.
# -----------------------------------------------------------------
model_rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced",
    random_state=42, n_jobs=-1
)
model_rf.fit(X_train, y_train)

# -----------------------------------------------------------------
# 4. Évaluation rapide sur le test set
# -----------------------------------------------------------------
pred = model_rf.predict(X_test)
proba = model_rf.predict_proba(X_test)[:, 1]

print("--- Modèle final : Random Forest + class_weight='balanced' ---")
print(f"Recall    : {recall_score(y_test, pred):.3f}")
print(f"Precision : {precision_score(y_test, pred):.3f}")
print(f"F1-score  : {f1_score(y_test, pred):.3f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, proba):.3f}")
print(f"PR-AUC    : {average_precision_score(y_test, proba):.3f}")
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, pred))
print("\n", classification_report(y_test, pred, target_names=["Pas de panne", "Panne"]))

# -----------------------------------------------------------------
# 5. Sauvegarde des artefacts nécessaires à predict.py
#    - model_final.joblib  : le modèle entraîné
#    - feature_cols.joblib : la liste ORDONNÉE des colonnes attendues en entrée
#      (indispensable pour que predict.py construise le vecteur de features
#      dans le bon ordre, sinon le modèle donnerait des résultats erronés)
# -----------------------------------------------------------------
joblib.dump(model_rf, MODEL_DIR + "model_final.joblib")
joblib.dump(feature_cols, MODEL_DIR + "feature_cols.joblib")
print("\nModèle final sauvegardé :", MODEL_DIR + "model_final.joblib")
print("Liste des features sauvegardée :", MODEL_DIR + "feature_cols.joblib")
