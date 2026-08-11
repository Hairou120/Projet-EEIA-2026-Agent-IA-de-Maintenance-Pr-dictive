import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

'''Le cœur de l'assistant RAG. Il gère :

La détection des intentions (salutations, identité, questions courtes).

La recherche des passages correspondants dans la base FAISS.

La formulation du prompt et l'appel au modèle local Ollama (Llama 3.2).'''

# Configuration des chemins d'accès
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

# Prompt Système strict pour forcer l'ancrage documentaire
PROMPT_TEMPLATE = """Vous êtes un expert technique en maintenance industrielle.
Répondez à la question en vous appuyant UNIQUEMENT sur le contexte extrait du manuel ci-dessous.
Si l'information n'est pas présente dans le contexte, répondez exactement : "Je ne trouve pas cette information dans le manuel technique."

CONTEXTE DU MANUEL :
{context}

QUESTION :
{question}

RÉPONSE DÉTAILLÉE :"""


class RAGService:
    def __init__(self):
        print("🔄 Chargement des embeddings et de l'index FAISS...")
        
        # 1. Chargement du modèle d'embeddings HuggingFace
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 2. Vérification de l'existence de la base vectorielle FAISS
        if not os.path.exists(VECTORSTORE_PATH):
            raise FileNotFoundError(
                f"❌ Index FAISS introuvable sous {VECTORSTORE_PATH}. "
                "Exécutez d'abord 'python scripts/build_vectorstore.py'."
            )
            
        # 3. Chargement de l'index FAISS local
        self.vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # 4. Connexion au LLM local Ollama (Llama 3.2)
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.1
        )
        
        # 5. Création du template de prompt
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        print("✅ Service RAG Ollama (Llama 3.2) initialisé avec succès !")

    def answer_question(self, question: str) -> dict:
        clean_q = question.strip().lower()
        
        # --- FILTRE 1 : Salutations ---
        greetings = ["bonjour", "bonsoir", "salut", "hello", "coucou", "hi"]
        if clean_q in greetings:
            return {
                "answer": "Bonjour ! Je suis votre assistant technique en maintenance prédictive. Posez-moi une question sur le manuel d'utilisation de votre équipement.",
                "sources": []
            }

        # --- FILTRE 2 : Remerciements et Clôture ---
        thanks = ["merci", "merci beaucoup", "super merci", "c'est parfait", "au revoir", "bye"]
        if clean_q in thanks:
            return {
                "answer": "Je vous en prie ! N'hésitez pas si vous avez d'autres questions sur le manuel technique.",
                "sources": []
            }

        # --- FILTRE 3 : Identité & Rôle ---
        identity_queries = ["qui es-tu", "qui es tu", "tu es qui", "que peux-tu faire", "comment tu t'appelles", "a quoi tu sers"]
        if any(query in clean_q for query in identity_queries):
            return {
                "answer": "Je suis un assistant virtuel de maintenance industrielle. Je suis conçu pour répondre à vos questions techniques à partir du manuel d'utilisation de vos équipements.",
                "sources": []
            }

        # --- FILTRE 4 : Requêtes très courtes ---
        if len(clean_q) < 3:
            return {
                "answer": "Votre question semble très courte. Veuillez formuler une phrase ou préciser le composant concerné.",
                "sources": []
            }

        # --- PIPELINE RAG (Recherche FAISS sans filtrage par score + Ollama) ---
        # 1. Recherche des 3 passages les plus proches dans FAISS
        docs = self.vectorstore.similarity_search(question, k=3)
        
        # 2. Assemblage des extraits en un seul bloc de contexte
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Formatage du prompt et génération de la réponse par Llama 3.2
        formatted_prompt = self.prompt.format(context=context_text, question=question)
        response = self.llm.invoke(formatted_prompt)
        answer_content = response.content if hasattr(response, "content") else str(response)
        
        # 4. Structuration des sources (Fichier PDF source + Numéro de Page)
        sources = []
        for doc in docs:
            page = doc.metadata.get("page", doc.metadata.get("page_number", 0)) + 1
            filename = doc.metadata.get("source_file", os.path.basename(doc.metadata.get("source", "Manuel")))
            sources.append({
                "file": filename,
                "page": page,
                "content_preview": doc.page_content[:150] + "..."
            })
            
        return {
            "answer": answer_content,
            "sources": sources
        }


# Pattern Singleton
_rag_service_instance = None

def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance