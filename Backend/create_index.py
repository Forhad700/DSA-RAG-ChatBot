import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "Knowledge_base")

FAISS_PATH = os.path.join(ROOT_DIR, "Faiss_index")

print(f"Checking data path: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"Error: Folder {DATA_PATH} not found!")
else:
    print("Loading documents...")

    pdf_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    txt_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = pdf_loader.load() + txt_loader.load()
    print(f"Loaded {len(documents)} documents")

    if len(documents) == 0:
        print("No documents found. Check your Knowledge_base folder.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        docs = text_splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks")

        print("Creating embeddings (this may take a while)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        db = FAISS.from_documents(docs, embeddings)

        os.makedirs(FAISS_PATH, exist_ok=True)
        db.save_local(FAISS_PATH)
        print(f"FAISS index created successfully at: {FAISS_PATH}")
