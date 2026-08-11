'''Définit la structure des données transmises à l'API 
(format Pydantic pour la question et la réponse).'''

from pydantic import BaseModel, Field
from typing import List

# Structure de la demande envoyée par l'utilisateur
class QueryData(BaseModel):
    # La question doit être une chaîne de caractères (str)
    question: str = Field(..., description="La question posée par l'utilisateur", example="Comment entretenir le moteur ?")


# Structure d'un document source retourné
class SourceDoc(BaseModel):
    page: int                   # Numéro de la page du PDF d'où provient l'information
    content_preview: str        # Extrait du texte de la page (aperçu)


# Structure globale de la réponse envoyée par l'API
class ChatResponse(BaseModel):
    answer: str                 # La réponse générée par Ollama
    sources: List[SourceDoc]    # La liste des pages PDF utilisées comme sources