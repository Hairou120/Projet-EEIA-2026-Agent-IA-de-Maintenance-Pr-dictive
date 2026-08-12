"""
training_models.py
Feature engineering, split train/test, entraînement de plusieurs modèles
candidats (Régression Logistique, Random Forest, Random Forest + SMOTE)
et évaluation comparée. Sert à choisir le modèle final avant train.py.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    ConfusionMatrixDisplay, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------
# 0. Configuration des chemins
# -----------------------------------------------------------------
BASE_DIR = r"E:\AKWPJ\EEEIA_2026\Projet_Agent_IA"
DATA_PATH = BASE_DIR + r"\ai4i2020.csv"
FEATURED_PATH = BASE_DIR + r"\ai4i2020_featured.csv"
MODEL_DIR = BASE_DIR + r"\models\\"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===================================================================
# PARTIE A — FEATURE ENGINEERING
# ===================================================================
# 1. Chargement des données brutes
df1 = pd.read_csv(DATA_PATH)
df = df1.copy()

# 2. Encodage ordinal de la variable catégorielle "Type"
#    Ordinal choisi (et pas one-hot) car l'EDA montre un ordre logique du
#    taux de panne : L (3.92%) > M (2.77%) > H (2.09%) -- confirmé
#    empiriquement meilleur que le one-hot (PR-AUC 0.877 vs 0.862).
type_map = {"L": 0, "M": 1, "H": 2}
df["Type_encoded"] = df["Type"].map(type_map)

# 3. Variables dérivées (feature engineering "métier")
# temp_diff : écart entre température procédé et température ambiante.
df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
# power_W : puissance mécanique en Watts = Couple [Nm] x Vitesse angulaire [rad/s].
# (rpm -> rad/s : x 2*pi/60, ce qui équivaut à x pi/30)
df["power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (np.pi / 30)
# strain : produit Couple x Usure outil, proxy de la contrainte mécanique cumulée.
df["strain"] = df["Torque [Nm]"] * df["Tool wear [min]"]

# Définition des colonnes finales du modèle
feature_cols = [
    "Type_encoded",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "temp_diff", "power_W", "strain",
]
target_col = "Machine failure"

# Sauvegarde du dataset enrichi
df.to_csv(FEATURED_PATH, index=False)
print("Feature engineering terminé. Dataset enrichi sauvegardé :", FEATURED_PATH)

# ===================================================================
# PARTIE B — SPLIT TRAIN/TEST ET ENTRAÎNEMENT DES MODÈLES
# ===================================================================
df = pd.read_csv(FEATURED_PATH)
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=8000, test_size=2000, stratify=y, random_state=42
)
print(f"Train : {X_train.shape[0]} lignes ({100*y_train.mean():.2f}% de pannes)")
print(f"Test  : {X_test.shape[0]} lignes ({100*y_test.mean():.2f}% de pannes)")

# Mise à l'échelle des données (nécessaire pour la régression logistique)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# MODELE A — Régression Logistique (baseline)
# -----------------------------------------------------------------
model_reglog = LogisticRegression(
    class_weight="balanced", random_state=42, max_iter=1000
)
model_reglog.fit(X_train_scaled, y_train)
pred_reglog = model_reglog.predict(X_test_scaled)

r_reglog = recall_score(y_test, pred_reglog)
p_reglog = precision_score(y_test, pred_reglog)
f_reglog = f1_score(y_test, pred_reglog)
auc_reglog = roc_auc_score(y_test, model_reglog.predict_proba(X_test_scaled)[:, 1])

print("\n--- Régression logistique ---")
print(f"Recall={r_reglog:.3f} | Precision={p_reglog:.3f} | F1={f_reglog:.3f} | AUC={auc_reglog:.3f}")
print("\n--- Evaluation de la Regression logistique ---\n",
      classification_report(y_test, pred_reglog, target_names=["Pas de panne", "Panne"]))

matrice = confusion_matrix(y_test, pred_reglog)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
ConfusionMatrixDisplay(matrice, display_labels=["Pas de panne", "Panne"]).plot(
    ax=axes, cmap="Greens", colorbar=False
)
axes.set_title("Matrice de confusion - Regression logistique")
plt.tight_layout()
plt.savefig(MODEL_DIR + "confusion_matrix_reglog.png")
plt.show()

# -----------------------------------------------------------------
# MODELE B — Random Forest + class_weight="balanced"
#    Entraîné sur les features BRUTES (non standardisées) : les arbres de
#    décision travaillent par seuils, pas par distance, donc la mise à
#    l'échelle n'apporte rien ici (contrairement à la régression logistique).
# -----------------------------------------------------------------
model_rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced",
    random_state=42, n_jobs=-1
)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

r_rf = recall_score(y_test, pred_rf)
p_rf = precision_score(y_test, pred_rf)
f_rf = f1_score(y_test, pred_rf)
auc_rf = roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1])

print("\n--- Random Forest ---")
print(f"Recall={r_rf:.3f} | Precision={p_rf:.3f} | F1={f_rf:.3f} | AUC={auc_rf:.3f}")
print("\n--- Evaluation du Random Forest ---\n",
      classification_report(y_test, pred_rf, target_names=["Pas de panne", "Panne"]))

matrice = confusion_matrix(y_test, pred_rf)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
ConfusionMatrixDisplay(matrice, display_labels=["Pas de panne", "Panne"]).plot(
    ax=axes, cmap="Greens", colorbar=False
)
axes.set_title("Matrice de confusion - Random Forest")
plt.tight_layout()
plt.savefig(MODEL_DIR + "confusion_matrix_rf.png")
plt.show()

# -----------------------------------------------------------------
# MODELE C — Random Forest + SMOTE (rééchantillonnage au lieu de pondération)
# -----------------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

model_rf_smote = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
model_rf_smote.fit(X_train_smote, y_train_smote)
pred_rf_smote = model_rf_smote.predict(X_test_scaled)

# CORRECTION : les métriques doivent être calculées sur pred_rf_smote,
# pas sur pred_rf (bug de copier-coller dans la version notebook).
r_rfsmote = recall_score(y_test, pred_rf_smote)
p_rfsmote = precision_score(y_test, pred_rf_smote)
f_rfsmote = f1_score(y_test, pred_rf_smote)
auc_rfsmote = roc_auc_score(y_test, model_rf_smote.predict_proba(X_test_scaled)[:, 1])

print("\n--- Random Forest + SMOTE ---")
print(f"Recall={r_rfsmote:.3f} | Precision={p_rfsmote:.3f} | F1={f_rfsmote:.3f} | AUC={auc_rfsmote:.3f}")
print("\n--- Evaluation du RandomForest + Smote ---\n",
      classification_report(y_test, pred_rf_smote, target_names=["Pas de panne", "Panne"]))

matrice = confusion_matrix(y_test, pred_rf_smote)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
ConfusionMatrixDisplay(matrice, display_labels=["Pas de panne", "Panne"]).plot(
    ax=axes, cmap="Greens", colorbar=False
)
axes.set_title("Matrice de confusion - Random Forest SMOTE")
plt.tight_layout()
plt.savefig(MODEL_DIR + "confusion_matrix_rf_smote.png")
plt.show()

# -----------------------------------------------------------------
# Tableau récapitulatif + sauvegarde des 3 modèles candidats
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'Modèle':<30}{'Recall':>10}{'Precision':>12}{'F1':>8}{'AUC':>8}")
print("=" * 70)
print(f"{'Régression Logistique':<30}{r_reglog:>10.3f}{p_reglog:>12.3f}{f_reglog:>8.3f}{auc_reglog:>8.3f}")
print(f"{'Random Forest':<30}{r_rf:>10.3f}{p_rf:>12.3f}{f_rf:>8.3f}{auc_rf:>8.3f}")
print(f"{'Random Forest + SMOTE':<30}{r_rfsmote:>10.3f}{p_rfsmote:>12.3f}{f_rfsmote:>8.3f}{auc_rfsmote:>8.3f}")
print("\n=> Random Forest (class_weight) retenu comme modèle final : meilleure")
print("   precision avec très peu de fausses alertes -> voir train.py")

joblib.dump(model_reglog, MODEL_DIR + "model_reglog.joblib")
joblib.dump(model_rf, MODEL_DIR + "model_rf.joblib")
joblib.dump(model_rf_smote, MODEL_DIR + "model_rf_smote.joblib")
joblib.dump(scaler, MODEL_DIR + "scaler.joblib")
joblib.dump(feature_cols, MODEL_DIR + "feature_cols.joblib")
print("\nModèles, scaler et feature_cols sauvegardés dans", MODEL_DIR)
