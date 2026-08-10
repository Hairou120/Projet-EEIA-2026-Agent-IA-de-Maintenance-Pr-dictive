from pydantic import BaseModel, Field
from typing import List, Union, Any


class QueryData(BaseModel):
    """Schéma de la requête envoyée par le technicien depuis l'interface Web."""
    question: str = Field(
        ...,
        min_length=3,
        description="La question posée par l'utilisateur à l'assistant RAG.",
        json_schema_extra={"example": "Que faire en cas d'oscillation anormale du couple moteur ?"}
    )


class SourceDetail(BaseModel):
    """Schéma décrivant un extrait de document ayant servi à la réponse."""
    id: int = Field(..., description="Numéro d'ordre de la source.")
    page: Union[int, str] = Field(..., description="Numéro de page dans le manuel PDF.")
    content_preview: str = Field(..., description="Aperçu textuel du chunk extrait.")


class ChatResponse(BaseModel):
    """Schéma de la réponse renvoyée au Frontend par la route /chat."""
    answer: str = Field(..., description="Réponse générée par le LLM (Llama 3.1 via Groq).")
    sources: List[SourceDetail] = Field(
        default=[],
        description="Liste des extraits du manuel utilisés comme contexte."
    )