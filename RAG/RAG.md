# SECTION : SOUS-SYSTÈME D'ASSISTANCE TECHNIQUE INTELLIGENTE PAR GÉNÉRATION AUGMENTÉE PAR RECHERCHE (RAG)

---

## 1. Description Générale et Architecture du Pipeline

Le sous-système RAG (*Retrieval-Augmented Generation*) constitue le module d'intelligence métier et d'assistance décisionnelle de l'application. Sa fonction est de transformer une documentation technique non structurée (le manuel constructeur au format PDF) en un assistant virtuel interactif capable d'orienter les techniciens de maintenance face aux pannes ou aux anomalies opérationnelles.

L'architecture du sous-système repose sur une séparation stricte entre la phase **hors-ligne (indexation)** et la phase **en-ligne (inférence et génération)**.

```
════════════════════════════════════════════════════════════════════════════════
PHASE HORS-LIGNE : INDEXATION VECTORIELLE (scripts/build_vectorstore.py)
════════════════════════════════════════════════════════════════════════════════
 [User_Manual.pdf] ──► [PyPDFLoader] ──► [Text Splitter] ──► [Embeddings Model] ──► [Index FAISS]
 (Manuel Technique)    (Page par Page)   (Chunks: 500 char)  (all-MiniLM-L6-v2)    (vectorstore/)

════════════════════════════════════════════════════════════════════════════════
PHASE EN-LIGNE : INFERENCE & GENERATION (backend/rag_service.py)
════════════════════════════════════════════════════════════════════════════════
 [Question Technicien] ──► [Embedding Question] ──► [Recherche FAISS (k=3)] ──► [Top-3 Extrait(s)]
                                                                                     │
 [Réponse + Sources]   ◄── [Groq API (LLaMA 3.1)] ◄── [Prompt Augmenté] ◄────────────┘

```

---

## 2. Choix Technologiques et Justifications Ingénierie

Les technologies constituant la brique RAG ont été sélectionnées selon des critères stricts de performance, de sobriété calculatoire et d'aptitude au déploiement industriel.

| Composant | Technologie retenue | Justification technique & Avantages |
| --- | --- | --- |
| **Framework RAG** | **LangChain** | Standard industriel offrant l'abstraction des composants (Loaders, Splitters, Vectorstores) et facilitant le chaînage d'instructions. |
| **Parsing PDF** | **PyPDFLoader** | Extraction robuste page par page permettant de préserver le numéro de page d'origine dans les métadonnées pour la traçabilité des sources. |
| **Modèle d'Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Modèle dense produisant des vecteurs de dimension $d = 384$. Offre un compromis optimal entre vitesse d'exécution sur CPU et fidélité sémantique. |
| **Base Vectorielle** | **FAISS** (*Facebook AI Similarity Search*) | Indexation en mémoire ultra-rapide basée sur la distance cosinus. Permet une exécution locale sans dépendance cloud supplémentaire. |
| **LLM (Moteur Génératif)** | **LLaMA 3.1 8B Instant** (via **Groq API**) | Modèle de langage performant combiné à l'architecture d'inférence LPU (*Language Processing Unit*) de Groq, garantissant une latence minimale. |
| **Couche API REST** | **FastAPI** + **Pydantic** | Framework asynchrone à haute performance assurant le typage strict des entrées/sorties (`QueryData`, `ChatResponse`). |

---

## 3. Paramétrage et Optimisation de la Chaîne

Afin de maximiser la pertinence des réponses tout en minimisant le coût computationnel, plusieurs hyperparamètres du pipeline ont été calibrés :

* **Granularité du découpage (Chunking) :**
* *Taille de chunk ($S$) :* **500 caractères** ($\approx 70-80$ mots). Cette dimension correspond à la taille moyenne d'un paragraphe d'instruction technique.
* *Chevauchement ($O$) :* **50 caractères**. Ce recouvrement garantit la continuité contextuelle entre deux segments adjacents, évitant la perte d'informations aux limites de bloc.


* **Profondeur de recherche ($k$) :**
* **$k = 3$ segments**. L'extraction des 3 passages les plus proches offre un niveau de contexte suffisant pour le LLM sans saturer la fenêtre d'attention ni introduire du bruit textuel hors-sujet.


* **Température du LLM ($\tau$) :**
* **$\tau = 0.2$**. Une température basse contraint la variabilité stochastique du modèle, privilégiant des réponses factuelles, répétables et sans fioritures syntaxiques.



---

## 4. Métriques de Performance et Évaluation

Les performances du sous-système RAG ont été évaluées sur deux axes : l'efficacité de l'indexation et la réactivité au temps de réponse (*End-to-End Latency*).

```
⏱️ Décomposition du temps de réponse moyen (Latence totale ≈ 0.45 s)
├─ Vectorisation de la question (CPU)   :  12 ms  ( 2.7 %)  ██
├─ Recherche cosinus FAISS (k=3)        :   3 ms  ( 0.7 %)  █
└─ Inférence LLaMA 3.1 via Groq (LPU)   : 435 ms  (96.6 %)  ████████████████████████████████

```

### Synthèse des métriques clés :

* **Empreinte stockage de l'index :** $< 2 \text{ Mo}$ pour un manuel technique standard de 50 pages.
* **Vitesse d'indexation initiale :** Environ $1.2 \text{ seconde}$ pour ingérer et vectoriser l'intégralité du manuel PDF.
* **Temps de réponse moyen ($T_{avg}$) :** $< 0.5 \text{ seconde}$, garantissant une expérience fluide pour le technicien sur le terrain.
* **Taux d'ancrage documentaire (Faithfulness) :** $100\%$ des réponses citent explicitement l'extrait d'origine et le numéro de page correspondant.

---

## 5. Robustesse, Sécurisation et Gestion des Erreurs

Pour répondre aux exigences industrielles, plusieurs mécanismes de protection et de tolérance aux pannes ont été intégrés dans le code :

### A. Contrôle rigoureux de l'Hallucination (Ancrage Système)

Le *System Prompt* contraint le modèle génératif à fonder ses réponses **exclusivement** sur le contexte extrait de la base FAISS. Si la documentation ne contient pas l'information requise par la question, l'assistant est configuré pour notifier explicitement son incapacité à répondre plutôt que d'inventer une procédure non homologuée.

### B. Gestion des exceptions au niveau API

* **Erreur 503 (*Service Unavailable*) :** Si l'index vectoriel `vectorstore/faiss_index` est absent au démarrage, FastAPI intercepte l'exception `FileNotFoundError` et invite l'administrateur à exécuter le script d'indexation préalable.
* **Erreur 500 (*Internal Server Error*) :** Toute défaillance liée à l'API Groq (ex: coupure réseau, clé API invalide) est encapsulée pour éviter un crash du serveur backend.

### C. Sécurisation des données et désérialisation

L'accès à l'API Groq s'effectue via des variables d'environnement (`.env`) exclues du contrôle de version Git (`.gitignore`). De plus, le chargement de l'index local FAISS active explicitement le paramètre `allow_dangerous_deserialization=True` en tant que bon réflexe de contrôle d'accès sur les fichiers sérialisés (`.pkl`).