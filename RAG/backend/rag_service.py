import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Chargement des variables d'environnement (ex: GROQ_API_KEY)
load_dotenv()

# Importations LangChain modernisées
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------------------------------------------------------
# 1. CONFIGURATION DES CHEMINS ET MODÈLES
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


# -----------------------------------------------------------------------------
# 2. CLASSE DU SERVICE RAG
# -----------------------------------------------------------------------------
class RAGService:
    def __init__(self):
        """
        Initialise le service RAG :
        1. Charge le modèle d'embeddings.
        2. Charge la base vectorielle FAISS depuis le disque.
        3. Initialise le client LLM (Groq API).
        """
        print("⚡ Initialisation du service RAG...")

        # A. Modèle d'embeddings (identique à celui utilisé lors de la vectorisation)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # B. Chargement de l'index FAISS
        if not os.path.exists(VECTORSTORE_PATH):
            raise FileNotFoundError(
                f"❌ L'index FAISS est introuvable dans '{VECTORSTORE_PATH}'. "
                "Exécutez d'abord le script 'scripts/build_vectorstore.py'."
            )

        self.vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True  # Requis pour charger un fichier pkl local
        )

        # Configuration du retriever (extrait les 3 segments les plus pertinents)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # C. Initialisation du LLM via Groq API
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("⚠️ Avertissement : GROQ_API_KEY n'est pas définie dans le fichier .env.")

        self.llm = ChatGroq(
            model_name=GROQ_MODEL_NAME,
            temperature=0.2,  # Température basse pour privilégier la précision technique
            groq_api_key=groq_api_key
        )

        # D. Template du Prompt Augmenté
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "Vous êtes un expert technique spécialisé dans la maintenance industrielle et l'assistance technique.\n"
             "Répondez à la question de l'utilisateur de manière claire, rigoureuse et structurée, "
             "en vous appuyant **exclusivement** sur le contexte fourni ci-dessous (issu du manuel technique).\n\n"
             "Consignes :\n"
             "1. Si le contexte ne contient pas l'information demandée, indiquez clairement que le manuel ne le précise pas.\n"
             "2. Si la question concerne une panne ou une alerte, indiquez les étapes de dépannage recommandées.\n"
             "3. Soignez la présentation avec des listes à puces si nécessaire.\n\n"
             "--- CONTEXTE DU MANUEL TECHNIQUE ---\n"
             "{context}\n"
             "------------------------------------"
            ),
            ("human", "{question}")
        ])

        print("✅ Service RAG prêt à recevoir des requêtes.")

    # -------------------------------------------------------------------------
    # 3. MÉTHODE D'INFERENCE (MOTEUR DE RECHERCHE + GÉNÉRATION)
    # -------------------------------------------------------------------------
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Exécute la chaîne RAG complète :
        1. Recherche des chunks pertinents dans FAISS.
        2. Formatage du contexte.
        3. Envoi du prompt au LLM.
        4. Extrait la réponse et les références de sources (pages).
        """
        # Étape 1 : Recherche vectorielle des k documents les plus proches
        retrieved_docs = self.retriever.invoke(question)

        # Étape 2 : Construction du contexte textuel et extraction des sources
        context_parts = []
        sources = []

        for i, doc in enumerate(retrieved_docs, start=1):
            page_num = doc.metadata.get("page", "Inconnue")
            source_content = doc.page_content.strip()

            context_parts.append(f"[Extrait {i} - Page {page_num}]\n{source_content}")
            sources.append({
                "id": i,
                "page": page_num,
                "content_preview": source_content[:150] + "..."  # Aperçu du texte
            })

        formatted_context = "\n\n".join(context_parts)

        # Étape 3 : Assemblage du Prompt et Appel du LLM
        prompt_messages = self.prompt_template.format_messages(
            context=formatted_context,
            question=question
        )

        response = self.llm.invoke(prompt_messages)

        # Étape 4 : Structuration du résultat
        return {
            "answer": response.content,
            "sources": sources
        }


# Instanciation d'un singleton pour réutilisation dans FastAPI
rag_service_instance = None

def get_rag_service() -> RAGService:
    global rag_service_instance
    if rag_service_instance is None:
        rag_service_instance = RAGService()
    return rag_service_instance