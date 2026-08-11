import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

'''Lit tous les PDF du dossier data/, découpe le texte en blocs (chunks), 
génère leurs empreintes mathématiques (embeddings) et crée la base vectorielle.'''

# Définition des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")


def build_vectorstore():
    print("🚀 Début de l'indexation globale des manuels PDF...")
    
    # 1. Vérification du dossier data
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        raise FileNotFoundError(f"❌ Dossier {DATA_DIR} créé, mais il est vide. Déposez-y vos manuels PDF.")

    # 2. Récupération de TOUS les fichiers PDF
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"❌ Aucun fichier PDF trouvé dans {DATA_DIR}")
    
    print(f"📚 {len(pdf_files)} manuels PDF détectés pour l'indexation.")
    
    all_documents = []
    
    # 3. Boucle de chargement sur chaque fichier PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"   📄 Chargement de : {pdf_file}")
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Ajout du nom du fichier dans les métadonnées pour savoir quelle machine est citée
        for doc in docs:
            doc.metadata["source_file"] = pdf_file
            
        all_documents.extend(docs)
    
    print(f"📖 Nombre total de pages chargées : {len(all_documents)}")
    
    # 4. Découpage du texte de tous les documents en sous-blocs (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_documents)
    print(f"🧩 Nombre total de blocs de texte (chunks) générés : {len(chunks)}")
    
    # 5. Conversion et stockage dans la base FAISS globale
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 6. Sauvegarde locale de l'index FAISS complet
    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"\n✅ Indexation réussie ! Base FAISS multi-machines créée sous {VECTORSTORE_PATH}")


if __name__ == "__main__":
    build_vectorstore()