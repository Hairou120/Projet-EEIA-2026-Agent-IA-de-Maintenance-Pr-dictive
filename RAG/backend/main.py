import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Importation des schémas Pydantic
from backend.schemas import QueryData, ChatResponse
# Importation du service RAG développé à l'étape 2
from backend.rag_service import get_rag_service

# -----------------------------------------------------------------------------
# INITIALISATION DE L'APPLICATION FASTAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API Maintenance Prédictive & Assistant RAG",
    description="Backend FastAPI fournissant l'assistance technique RAG basée sur la documentation PDF.",
    version="1.0.0"
)

# Configuration CORS (permet les requêtes depuis le frontend Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, restreindre aux URL autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# ENDPOINTS / ROUTES HTTP
# -----------------------------------------------------------------------------
@app.get("/health", tags=["Santé"])
def health_check():
    """Vérifie que l'API est en ligne."""
    return {"status": "online", "message": "API RAG opérationnelle"}


@app.post("/chat", response_model=ChatResponse, tags=["Assistant RAG"])
def chat_endpoint(query: QueryData):
    """
    Endpoint principal pour l'assistant RAG :
    1. Reçoit la question du technicien.
    2. Interroge le service RAG (FAISS + Groq LLM).
    3. Renvoie la réponse formatée et les références de sources.
    """
    try:
        # Récupération de l'instance du service RAG
        rag_service = get_rag_service()
        
        # Exécution de la chaîne RAG
        result = rag_service.answer_question(query.question)
        
        return result

    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf_err)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du serveur RAG : {str(e)}"
        )