"""
predict.py
Fonction de prédiction prête à être importée par le backend (FastAPI ou autre).
Prend les mesures BRUTES d'une machine (telles que fournies par les capteurs
ou saisies via un formulaire web) et renvoie la prédiction + la probabilité
de panne.

Exemple d'utilisation côté API (FastAPI) :

    from predict import predict_failure

    @app.post("/predict_failure")
    def predict_failure_endpoint(data: MachineData):
        return predict_failure(
            product_type=data.product_type,
            air_temperature_k=data.air_temperature_k,
            process_temperature_k=data.process_temperature_k,
            rotational_speed_rpm=data.rotational_speed_rpm,
            torque_nm=data.torque_nm,
            tool_wear_min=data.tool_wear_min,
        )
"""
import os
import numpy as np
import joblib

# -----------------------------------------------------------------
# Configuration des chemins et chargement des artefacts UNE SEULE FOIS
# au démarrage du service (pas à chaque appel : recharger le modèle à
# chaque requête serait beaucoup trop lent pour une API en production).
# -----------------------------------------------------------------
BASE_DIR = r"E:\AKWPJ\EEEIA_2026\Projet_Agent_IA"
MODEL_DIR = BASE_DIR + r"\models\\"

MODEL = joblib.load(os.path.join(MODEL_DIR, "model_final.joblib"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Seuil de décision. 0.5 par défaut (standard scikit-learn). Peut être
# abaissé (ex: 0.3) pour privilégier le Recall (moins de pannes ratées,
# au prix de plus de fausses alertes) selon la politique de l'usine.
SEUIL_DECISION = 0.5


def predict_failure(
    product_type: str,
    air_temperature_k: float,
    process_temperature_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> dict:
    """
    Renvoie une prédiction de panne à partir des mesures brutes d'une machine.

    Paramètres
    ----------
    product_type : "L", "M" ou "H" (qualité du produit)
    air_temperature_k : température ambiante, en Kelvin
    process_temperature_k : température du procédé, en Kelvin
    rotational_speed_rpm : vitesse de rotation, en tours/minute
    torque_nm : couple, en Newton-mètre
    tool_wear_min : usure de l'outil, en minutes cumulées

    Retour
    ------
    dict avec les clés :
        panne_predite      (bool)  : True si une panne est prédite
        probabilite_panne  (float) : probabilité de panne, entre 0 et 1
        niveau_risque      (str)   : "faible", "élevé" ou "critique"
    """
    # 1. Recalcul des mêmes features dérivées qu'à l'entraînement (train.py).
    #    IMPORTANT : ces formules doivent rester identiques à celles de
    #    train.py, sinon le modèle reçoit des données incohérentes.
    type_encoded = TYPE_MAP.get(product_type.upper(), 1)  # défaut "M" si type inconnu/absent
    temp_diff = process_temperature_k - air_temperature_k
    power_w = torque_nm * rotational_speed_rpm * (np.pi / 30)
    strain = torque_nm * tool_wear_min

    # 2. Construction du vecteur de features DANS LE MÊME ORDRE que feature_cols
    valeurs = {
        "Type_encoded": type_encoded,
        "Air temperature [K]": air_temperature_k,
        "Process temperature [K]": process_temperature_k,
        "Rotational speed [rpm]": rotational_speed_rpm,
        "Torque [Nm]": torque_nm,
        "Tool wear [min]": tool_wear_min,
        "temp_diff": temp_diff,
        "power_W": power_w,
        "strain": strain,
    }
    x = np.array([[valeurs[col] for col in FEATURE_COLS]])

    # 3. Prédiction (pas de scaler : le Random Forest a été entraîné sur
    #    les features brutes, non standardisées -- cf. train.py)
    proba_panne = float(MODEL.predict_proba(x)[0, 1])
    panne_predite = proba_panne >= SEUIL_DECISION

    return {
        "panne_predite": panne_predite,
        "probabilite_panne": round(proba_panne, 4),
        "seuil_utilise": SEUIL_DECISION,
        "niveau_risque": (
            "critique" if proba_panne >= 0.7 else
            "élevé" if proba_panne >= SEUIL_DECISION else
            "faible"
        ),
    }


if __name__ == "__main__":
    # Test rapide en local (exemple proche d'une zone à risque identifiée en EDA :
    # couple élevé + vitesse faible -> zone de surcharge mécanique)
    resultat = predict_failure(
        product_type="L",
        air_temperature_k=298.5,
        process_temperature_k=309.0,
        rotational_speed_rpm=1350,
        torque_nm=65.0,
        tool_wear_min=200,
    )
    print("Exemple de prédiction :", resultat)
