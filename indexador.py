import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import fitz
import os

def extract_text_from_pdfs(pdf_folder):
    documents = []
    
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            doc_path = os.path.join(pdf_folder, file)
            doc = fitz.open(doc_path)

            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            
            documents.append({"filename": file, "content": text})
    
    return documents

# Configuração do ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="procurement_articles", embedding_function=embedding_function)

# Processar PDFs e indexar uma única vez
pdf_folder = "C:/Users/Administrador/Downloads/Procurement-_AI_Agent/data"
pdf_data = extract_text_from_pdfs(pdf_folder)

# Adicionar documentos à coleção (apenas se ainda não foram indexados)
if collection.count() == 0:  
    for idx, doc in enumerate(pdf_data):
        collection.add(
            ids=[str(idx)], 
            documents=[doc["content"]],
            metadatas=[{"filename": doc["filename"]}]
        )
    print("Artigos indexados com sucesso! ✅")
else:
    print("Os artigos já estão indexados.")
