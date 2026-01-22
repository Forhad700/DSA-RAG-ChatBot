import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# RAG imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

FAISS_PATH = "../Faiss_index"

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS index...")
db = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",  
    temperature=0.7
)

SYSTEM_PROMPT = """
You are a helpful assistant.
Use the context to answer the question.
If you don't know, say you don't know.
Context: {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

app = FastAPI()

class QueryRequest(BaseModel):
    text: str

@app.post("/query")
def query_rag(req: QueryRequest):
    result = rag_chain.invoke({"input": req.text})
    return {"answer": result["answer"]}

@app.get("/")
def root():
    return {"message": "RAG API is running"}
