# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:37:56 2026

@author: HP
"""
import pandas as pd
import numpy as np
#chargement des données 
df=pd.read_csv("E:\\AKWPJ\\EEEIA_2026\\Projet_Agent_IA\\ai4i2020.csv")
base=df.copy()

#Taille des données
nb=base.shape
print("Dimensions de notre data_set:\n",nb)

#Type de données
base_types=base.dtypes
print(base_types)

#Affichage des 10 premieres lignes
base_10=base.head()
print(base)

#Vérification des valeurs manquantes
nb_values_isna=base.isna().sum()
print("compte des valeurs manquantes" ,nb_values_isna)

#Vérification de doublons
nb_doublons=base.duplicated().sum()
print(nb_doublons)
nb_doublons_Product_ID=base["Product ID"].duplicated()
print(nb_doublons_Product_ID)

#Statistique descriptive
print("\nStatistique descriptive"+"-"*20+"\n")
print(base["Machine failure"].value_counts())
#base.describe().T
num_columns=["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
desc=base[num_columns].describe().T
#Mesure de l'assymetrie de la distribution
desc["skewness"]= base[num_columns].skew()
desc["Kurtosis"]=base[num_columns].kurt()
print(desc)

for col in num_columns:
    Q1, Q3 = base[col].quantile(0.25), base[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_outliers = ((base[col] < lower) | (base[col] > upper)).sum()
    print(f"{col:30s}: {n_outliers:4d} outliers ({100*n_outliers/len(df):.2f}%)  "
          f"[bornes: {lower:.1f} - {upper:.1f}]")



#Matrice de correlation
base_correlation= base[num_columns+["Machine failure"]].corr()
print(base_correlation)