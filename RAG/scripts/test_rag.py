import os
import sys

# Ajout du dossier racine du projet au système d'importation Python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from backend.rag_service import get_rag_service


def main():
    print("=" * 65)
    print("🧪 TEST INTERACTIF DU SERVICE RAG (SANS FASTAPI)")
    print("=" * 65)

    # 1. Initialisation du service RAG
    try:
        print("\n⏳ Chargement du service RAG (FAISS + Embeddings + Groq)...")
        rag_service = get_rag_service()
        print("✅ Service RAG chargé avec succès !\n")
    except Exception as e:
        print(f"❌ Erreur critique lors de l'initialisation : {e}")
        print("💡 Vérifiez que :")
        print("   1. Le fichier '.env' contient bien votre GROQ_API_KEY.")
        print("   2. L'index FAISS existe dans 'vectorstore/faiss_index' (exécutez 'python scripts/build_vectorstore.py').")
        sys.exit(1)

    print(" tapez votre question ci-dessous (ou 'q' / 'exit' pour quitter).\n")

    # 2. Boucle d'interaction en ligne de commande
    while True:
        try:
            user_query = input("🗣️  Votre question : ").strip()

            # Conditions de sortie
            if user_query.lower() in ["q", "exit", "quit", ""]:
                print("\n👋 Fermeture du test RAG. À bientôt !")
                break

            print("\n🔍 Extraction FAISS & Génération de la réponse via Groq...")
            
            # Appel direct à la méthode du service RAG
            result = rag_service.answer_question(user_query)

            # Affichage de la réponse du LLM
            print("\n" + "─" * 65)
            print("🤖 RÉPONSE DE L'ASSISTANT RAG :")
            print("─" * 65)
            print(result["answer"])

            # Affichage des sources consultées
            print("\n" + "─" * 65)
            print("📚 SOURCES EXTRAITES DU MANUEL TECHNIQUE :")
            print("─" * 65)
            
            sources = result.get("sources", [])
            if sources:
                for src in sources:
                    print(f"• [Extrait #{src['id']}] - Page {src['page']}")
                    print(f"  Aperçu : \"{src['content_preview']}\"\n")
            else:
                print("  Aucune source spécifique extraite.\n")

            print("=" * 65 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interruption par l'utilisateur. Quitter.")
            break
        except Exception as e:
            print(f"\n❌ Une erreur est survenue lors du traitement : {e}\n")


if __name__ == "__main__":
    main()