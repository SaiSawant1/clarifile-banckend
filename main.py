import os
import tempfile
import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChatPDF API",
    description="API for chatting with PDF documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your Next.js app origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

active_sessions = {}

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    session_id: str

@app.post("/upload-pdf", response_model=dict)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    Returns a session ID for future queries.
    """
    try:
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, file.filename)
        
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"PDF saved to: {pdf_path}")
        
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        embedding = HuggingFaceEmbeddings()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text = text_splitter.split_documents(documents)
        
        db = Chroma.from_documents(text, embedding)
        llm = HuggingFaceHub(
            repo_id="OpenAssistant/oasst-sft-1-pythia-12b", 
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3})
        )
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "chain": retrieval_chain,
            "pdf_path": pdf_path,
            "filename": file.filename
        }
        return {
            "success": True,
            "message": "Document successfully processed",
            "session_id": session_id,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a previously uploaded PDF document using its session ID.
    """
    try:
        session_id = request.session_id
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")
        session = active_sessions[session_id]
        chain = session["chain"]
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        logger.info(f"Processing query for session {session_id}: {query}")
        answer = chain.run(query)
        return {
            "answer": answer,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/sessions/{session_id}", response_model=dict)
async def get_session(session_id: str):
    """
    Get information about a specific session.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    return {
        "session_id": session_id,
        "filename": session["filename"],
        "active": True
    }

@app.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(session_id: str):
    """
    Delete a session and clean up resources.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clean up PDF file
    try:
        os.remove(active_sessions[session_id]["pdf_path"])
    except Exception as e:
        logger.warning(f"Could not remove PDF file: {str(e)}")
    
    # Remove session
    del active_sessions[session_id]
    
    return {"success": True, "message": "Session deleted successfully"}

@app.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "active_sessions": len(active_sessions)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
