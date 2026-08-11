import sys
import os

'''Script de test en ligne de commande pour interroger l'assistant RAG 
directement dans le terminal.'''



# Ajout du dossier parent au système de modules Python pour permettre l'importation de 'backend'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_service import get_rag_service


def test_cli():
    print("=== TEST INTERACTIF DU MODULE RAG (OLLAMA / LLAMA3.2) ===")
    try:
        # Initialisation du service
        rag = get_rag_service()
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return

    # Boucle d'interaction infinie jusqu'à la saisie de 'q'
    while True:
        q = input("\n🗣️ Votre question (ou 'q' pour quitter) : ").strip()
        if q.lower() == 'q':
            print("👋 Fermeture du test.")
            break
        if not q:
            continue
        
        print("⌛ Recherche et génération en cours...")
        # Appel du service RAG
        res = rag.answer_question(q)
        
        # Affichage du résultat
        print(f"\n💡 Réponse :\n{res['answer']}")
        print("\n📌 Sources :")
        for s in res['sources']:
            print(f"  - Page {s['page']} : {s['content_preview']}")


if __name__ == "__main__":
    test_cli()