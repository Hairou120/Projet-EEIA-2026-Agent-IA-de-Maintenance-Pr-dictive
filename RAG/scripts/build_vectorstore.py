import os
import sys
from dotenv import load_dotenv

# Chargement des variables d'environnement (.env)
load_dotenv()

# Importations LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Chemins des fichiers (relatifs au dossier RAG)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "User_Manual.pdf")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

# Nom du modèle d'embeddings open-source
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore():
    print("=" * 60)
    print("🚀 DÉMARRAGE DU PIPELINE D'INDEXATION RAG")
    print("=" * 60)

    # 1. Vérification de l'existence du manuel PDF
    if not os.path.exists(PDF_PATH):
        print(f"❌ Erreur : Le fichier PDF est introuvable sous '{PDF_PATH}'.")
        print("Veuillez vérifier que 'User_Manual.pdf' est bien dans 'RAG/data/'.")
        sys.exit(1)

    # 2. Chargement du document PDF
    print(f"\n📄 [1/4] Chargement du document PDF : {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"   └─ {len(documents)} page(s) chargée(s) avec succès.")

    # 3. Découpage du texte en segments (Chunks)
    print("\n✂️  [2/4] Découpage du texte en chunks (segments)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Taille max de chaque segment (en caractères)
        chunk_overlap=50,      # Chevauchement pour conserver le contexte
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   └─ {len(chunks)} chunks générés au total.")

    # 4. Initialisation du modèle d'Embeddings
    print(f"\n🧠 [3/4] Initialisation du modèle d'embeddings ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 5. Création de l'index vectoriel FAISS et sauvegarde
    print("\n📦 [4/4] Vectorisation des chunks et création du magasin FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Assure que le dossier de destination existe
    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    
    # Sauvegarde locale sur disque
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"   └─ Index FAISS sauvegardé avec succès dans : {VECTORSTORE_PATH}")

    print("\n" + "=" * 60)
    print("✅ INDEXATION TERMINÉE AVEC SUCCÈS ! La base est prête.")
    print("=" * 60)


if __name__ == "__main__":
    build_vectorstore()