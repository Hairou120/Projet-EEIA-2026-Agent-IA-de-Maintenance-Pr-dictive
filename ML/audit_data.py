"""
audit_data.py
Étapes 1 et 2 — Audit des données, analyse statistique exploratoire (EDA)
et visualisation.
Objectif : vérifier la fiabilité du dataset avant toute modélisation, puis
comprendre statistiquement les variables et leur lien avec la cible, pour
justifier les décisions prises dans training_models.py.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------------------------------------------
# 0. Configuration des chemins (à adapter si le dossier projet change)
# -----------------------------------------------------------------
BASE_DIR = r"E:\AKWPJ\EEEIA_2026\Projet_Agent_IA"
DATA_PATH = BASE_DIR + r"\ai4i2020.csv"

# -----------------------------------------------------------------
# 1. Chargement des données brutes
# -----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

num_cols = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]

# ===================================================================
# ÉTAPE 1 — AUDIT STRUCTUREL
# But : détecter tout problème de qualité de données AVANT de modéliser.
# ===================================================================
print("=" * 60)
print("1. AUDIT STRUCTUREL")
print("=" * 60)
print("Dimensions :", df.shape)

print("\nTypes de données :")
print(df.dtypes)

# isna().sum() compte les valeurs manquantes par colonne. Un modèle ne peut
# pas s'entraîner sur des valeurs manquantes non traitées.
print("\nValeurs manquantes par colonne :")
print(df.isna().sum())

# duplicated() détecte les lignes strictement identiques. Des doublons
# fausseraient l'apprentissage en sur-représentant certains cas.
print("\nDoublons (lignes complètes) :", df.duplicated().sum())
print("Doublons Product ID :", df["Product ID"].duplicated().sum())

# Vérifie que UDI est bien un identifiant séquentiel propre (1 à 10000),
# confirme l'intégrité du fichier (pas de lignes manquantes/corrompues).
udi_ok = (df["UDI"] == range(1, len(df) + 1)).all()
print("UDI unique et séquentiel de 1 à 10000 ?", udi_ok)

# ===================================================================
# ÉTAPE 2 — STATISTIQUES DESCRIPTIVES
# But : comprendre la forme de chaque distribution avant de décider
# des transformations à appliquer (feature engineering).
# ===================================================================
print("\n" + "=" * 60)
print("2. STATISTIQUES DESCRIPTIVES (variables numériques)")
print("=" * 60)
desc = df[num_cols].describe().T
# skewness : mesure l'asymétrie de la distribution (0 = symétrique comme une
# loi normale, >0 = étalée à droite, <0 = étalée à gauche).
desc["skewness"] = df[num_cols].skew()
# kurtosis : mesure le poids des valeurs extrêmes (queues de distribution).
# Une valeur élevée signale des valeurs extrêmes plus fréquentes qu'une loi normale.
desc["kurtosis"] = df[num_cols].kurt()
print(desc)

# ===================================================================
# ÉTAPE 2 (suite) — DÉTECTION D'OUTLIERS PAR LA MÉTHODE IQR
# But : repérer les valeurs extrêmes, PUIS vérifier si elles sont du bruit
# ou un signal prédictif avant de décider de les garder/traiter.
# ===================================================================
print("\n" + "=" * 60)
print("3. DÉTECTION D'OUTLIERS (méthode IQR)")
print("=" * 60)
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col:30s}: {n_outliers:4d} outliers ({100*n_outliers/len(df):.2f}%)  "
          f"[bornes: {lower:.1f} - {upper:.1f}]")

# Vérification cruciale : ces outliers sont-ils informatifs (liés à la panne)
# ou de simples erreurs de mesure ? On compare le taux de panne DANS et HORS
# outliers pour chaque variable clé.
print("\n" + "=" * 60)
print("4. LES OUTLIERS SONT-ILS INFORMATIFS OU DU BRUIT ?")
print("=" * 60)
taux_global = df["Machine failure"].mean()
print(f"Taux de panne global (référence) : {100*taux_global:.2f}%\n")
for col in ["Rotational speed [rpm]", "Torque [Nm]"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask_out = (df[col] < lower) | (df[col] > upper)
    taux_out = df.loc[mask_out, "Machine failure"].mean()
    taux_in = df.loc[~mask_out, "Machine failure"].mean()
    ratio = taux_out / taux_global
    print(f"{col}:")
    print(f"  Taux de panne DANS les outliers : {100*taux_out:.2f}%")
    print(f"  Taux de panne HORS outliers     : {100*taux_in:.2f}%")
    print(f"  => Risque de panne x{ratio:.1f} par rapport à la moyenne\n")
# Conclusion (justifie la décision prise dans training_models.py) :
# les outliers sont fortement informatifs (risque x26 sur Torque) -> on ne
# les supprime PAS, contrairement à une pratique de nettoyage "par défaut".

# ===================================================================
# ÉTAPE 2 (suite) — MATRICE DE CORRÉLATION
# But : détecter la multicolinéarité (variables redondantes) et identifier
# les variables les plus liées à la cible.
# ===================================================================
print("=" * 60)
print("5. MATRICE DE CORRÉLATION (variables numériques + cible)")
print("=" * 60)
corr_matrix = df[num_cols + ["Machine failure"]].corr()
print(corr_matrix.round(3))
# Constat clé : Air/Process temperature corrélées à 0.88, et Rotational speed/
# Torque corrélées à -0.88 -> justifie la création de temp_diff et power_W
# dans training_models.py (capturer l'écart utile plutôt que la redondance).

# ===================================================================
# ÉTAPE 2 (suite) — ANALYSE PAR SOUS-TYPE DE PANNE
# But : comprendre la composition de la cible Machine failure, et repérer
# le risque de data leakage si ces colonnes étaient utilisées comme features.
# ===================================================================
print("\n" + "=" * 60)
print("6. ANALYSE PAR SOUS-TYPE DE PANNE")
print("=" * 60)
for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
    n = df[col].sum()
    print(f"{col}: {n} occurrences ({100*n/len(df):.2f}%)")
print(f"\nSomme des sous-pannes : {df[['TWF','HDF','PWF','OSF','RNF']].sum().sum()}")
print(f"Machine failure total : {df['Machine failure'].sum()}")
overlap = df[(df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1) > 1)]
print(f"Lignes avec plusieurs causes de panne simultanées : {len(overlap)}")

# ===================================================================
# ÉTAPE 2 (suite) — RÉPARTITION PAR TYPE DE PRODUIT
# But : vérifier si la qualité du produit (L/M/H) influence le taux de panne,
# pour justifier l'encodage ordinal choisi dans training_models.py.
# ===================================================================
print("\n" + "=" * 60)
print("7. RÉPARTITION PAR TYPE DE PRODUIT")
print("=" * 60)
print(df["Type"].value_counts())
print("\nTaux de panne par Type :")
print(df.groupby("Type")["Machine failure"].mean().sort_values(ascending=False))

# ===================================================================
# ÉTAPE 2 (suite) — ANALYSE DE VARIANCE (ANOVA)
# But : tester STATISTIQUEMENT si la moyenne de chaque variable diffère
# significativement entre le groupe "panne" et le groupe "pas de panne".
# ===================================================================
print("\n" + "=" * 60)
print("8. ANALYSE DE VARIANCE (ANOVA) — Machine failure vs variables")
print("=" * 60)
anova_results = []
for col in num_cols:
    groupe_ok = df.loc[df["Machine failure"] == 0, col]
    groupe_panne = df.loc[df["Machine failure"] == 1, col]
    f_stat, p_value = stats.f_oneway(groupe_ok, groupe_panne)
    anova_results.append({"variable": col, "F_statistique": f_stat, "p_value": p_value})

anova_df = pd.DataFrame(anova_results).sort_values("F_statistique", ascending=False)
anova_df["significatif (p<0.05)"] = anova_df["p_value"] < 0.05
print(anova_df.to_string(index=False))

# ===================================================================
# VISUALISATIONS
# ===================================================================
df["Failure_label"] = df["Machine failure"].map({0: "Pas de panne", 1: "Panne"})
sns.set_theme(style="whitegrid")

# 1. Distributions par classe (boxplots)
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, col in zip(axes, num_cols):
    sns.boxplot(data=df, x="Failure_label", y=col, ax=ax, hue="Failure_label",
                palette={"Pas de panne": "#4C72B0", "Panne": "#C44E52"}, legend=False)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(BASE_DIR + r"\eda_boxplots_par_classe.png", dpi=120)
plt.close()

# 2. Matrice de corrélation (heatmap)
plt.figure(figsize=(7, 6))
corr = df[num_cols + ["Machine failure"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig(BASE_DIR + r"\eda_correlation_heatmap.png", dpi=120)
plt.close()

# 3. Torque vs Rotational speed, colorié par panne
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="Rotational speed [rpm]", y="Torque [Nm]",
    hue="Failure_label", palette={"Pas de panne": "#4C72B0", "Panne": "#C44E52"},
    alpha=0.6, s=25
)
plt.title("Torque vs Vitesse de rotation — zones à risque")
plt.tight_layout()
plt.savefig(BASE_DIR + r"\eda_scatter_torque_speed.png", dpi=120)
plt.close()

# 4. Taux de panne par type de produit
plt.figure(figsize=(6, 5))
taux = df.groupby("Type")["Machine failure"].mean().sort_values(ascending=False) * 100
sns.barplot(x=taux.index, y=taux.values, hue=taux.index, palette="Blues_d", legend=False)
plt.ylabel("Taux de panne (%)")
plt.title("Taux de panne par qualité de produit")
plt.tight_layout()
plt.savefig(BASE_DIR + r"\eda_taux_panne_par_type.png", dpi=120)
plt.close()

# 5. ANOVA — pouvoir discriminant de chaque variable
anova_plot = anova_df.sort_values("F_statistique", ascending=True)
plt.figure(figsize=(8, 5))
plt.barh(anova_plot["variable"], anova_plot["F_statistique"], color="#55A868")
plt.xlabel("F-statistique (ANOVA)")
plt.title("Pouvoir discriminant de chaque variable (ANOVA vs Machine failure)")
plt.tight_layout()
plt.savefig(BASE_DIR + r"\eda_anova_fstat.png", dpi=120)
plt.close()

# 6. Diagramme circulaire — répartition Machine failure
counts = df["Machine failure"].value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(
    counts, labels=["Pas de panne", "Panne"], colors=["#4C72B0", "#C44E52"],
    autopct="%1.2f%%", startangle=90, explode=(0, 0.12), textprops={"fontsize": 11}
)
plt.title(f"Répartition de Machine failure (n={len(df)})")
plt.tight_layout()
plt.savefig(BASE_DIR + r"\eda_pie_machine_failure.png", dpi=120)
plt.close()

print("\n6 graphiques générés avec succès dans", BASE_DIR)
