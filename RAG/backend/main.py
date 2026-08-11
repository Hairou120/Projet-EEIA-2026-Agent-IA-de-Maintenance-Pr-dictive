from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import QueryData, ChatResponse
from backend.rag_service import get_rag_service

'''Serveur FastAPI qui transforme le service RAG en API web réutilisable
par l'interface graphique.'''

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API RAG Maintenance Prédictive",
    description="API d'assistance technique locale via Ollama",
    version="1.0.0"
)

# Configuration CORS pour autoriser l'interface Streamlit (Web) à communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Autorise toutes les origines
    allow_credentials=True,     # Autorise l'envoi de cookies/headers authentifiés
    allow_methods=["*"],        # Autorise toutes les méthodes HTTP (GET, POST, etc.)
    allow_headers=["*"],        # Autorise tous les entêtes
)

# Endpoint de vérification de l'état de l'API (Healthcheck)
@app.get("/health", tags=["Santé"])
def health_check():
    return {"status": "online", "message": "API RAG opérationnelle"}

# Endpoint principal du Chat : reçoit la question et renvoie la réponse
@app.post("/chat", response_model=ChatResponse, tags=["Assistant RAG"])
def chat_endpoint(payload: QueryData):
    try:
        # Récupération de l'instance RAG
        rag_service = get_rag_service()
        # Traitement de la question et génération de la réponse
        result = rag_service.answer_question(payload.question)
        return result
    except Exception as e:
        # En cas d'erreur interne, renvoie un code HTTP 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne RAG : {str(e)}"
        )