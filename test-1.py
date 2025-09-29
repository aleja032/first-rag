import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader  # Loader para PDFs
import getpass
import os
from uuid import uuid4

def main():
    # 1. Inicializar embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Crear o cargar vector store con persistencia
    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_db"  # Carpeta para guardar datos
    )

    
    LANGSMITH_TRACING="true"
    LANGSMITH_API_KEY="AIzaSyBRZls5DQHo4bX1Gp4cq26icok-McxVJn4"

#-------------------------------------------------------------------------------
    # # 3. Cargar documento desde archivo
    # with open("QA_Problems.pdf", "r", encoding="utf-8") as f:
    #     content = f.read()
    # doc = Document(page_content=content)

    # # 4. Preparar lista de documentos
    # docs = [doc]

    # # 5. Dividir documentos en fragmentos
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # chunks = []
    # for doc in docs:
    #     chunks.extend(text_splitter.split_text(doc.page_content))

    # # 6. Agregar fragmentos a ChromaDB
    # vector_store.add_documents([Document(page_content=chunk) for chunk in chunks])

    # # 7. Crear retriever
    # retriever = vector_store.as_retriever()

    # 8. Inicializar LLM Gemini con API key desde variable de entorno
#-------------------------------------------------------------------------------
 # 3. Cargar PDF y extraer documentos (una página por Document)
    pdf_files = ["QA_Problems.pdf", "design-patterns-es-demo.pdf"]
    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()  # Lista de Document, uno por página
        all_docs.extend(docs)
    

   #-------------------------------------------------------------------------------
    #load only one file
    # loader = PyPDFLoader("QA_Problems.pdf")
    # docs = loader.load()  # Lista de Document, uno por página

    # 4. Dividir documentos en fragmentos para mejor indexación
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
   #-------------------------------------------------------------------------------

    documents_with_ids = [
        Document(page_content=chunk.page_content, metadata=chunk.metadata, id=str(uuid4()))
        for chunk in chunks
    ]

    # Verificar los documentos cargados
    # print(f"Cargados {len(all_docs)} documentos desde los PDFs.")
    # collection = vector_store._collection  # Acceso directo a la colección interna
    # all_docs = collection.get(include=["documents", "metadatas"])
    # print(all_docs["documents"])  # Lista de textos almacenados
   #-------------------------------------------------------------------------------

    # # 5. Agregar fragmentos a ChromaDB
    vector_store.add_documents(documents_with_ids)

    # 6. Crear retriever
    retriever = vector_store.as_retriever()
    #-------------------------------------------------------------------------------

    #7
    api_key = os.getenv("GOOGLE_API_KEY")
    print("API Key:", api_key)
    llm = init_chat_model("google_genai:gemini-2.0-flash", api_key="AIzaSyBRZls5DQHo4bX1Gp4cq26icok-McxVJn4")

    # 8. Crear cadena RetrievalQA que combina retriever y LLM
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 9. Realizar consulta y obtener respuesta generada
    query = "tell me about emails problems "
    respuesta = qa_chain.run(query)

    # 10. Mostrar respuesta generada
    print("Respuesta generada por Gemini:")
    print(respuesta)

if __name__ == "__main__":
    main()
