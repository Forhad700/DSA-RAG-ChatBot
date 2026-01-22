import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Folder where PDFs/TXT files are
DATA_PATH = "../Knowledge_base"

# Folder where FAISS index will be saved
FAISS_PATH = "../Faiss_index"

print("Loading documents...")

# Load all PDFs
pdf_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

# Load all TXT files
txt_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

# Combine all documents
documents = pdf_loader.load() + txt_loader.load()
print(f"Loaded {len(documents)} documents")

# Split documents into chunks for embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

# Create embeddings
print("Creating embeddings (this may take a while)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS index
db = FAISS.from_documents(docs, embeddings)

# Save index
os.makedirs(FAISS_PATH, exist_ok=True)
db.save_local(FAISS_PATH)
print("FAISS index created successfully!")
