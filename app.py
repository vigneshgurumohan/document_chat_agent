from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
import os
import shutil
from pathlib import Path
import json
from typing import List, Optional
import logging

from document_agent import DocumentChatAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Document Chat Agent", version="1.0.0")

# Create necessary directories
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global agent instance
agent = None

# Initialize agent
def initialize_agent():
    global agent
    try:
        # Get OpenAI API key from environment variable
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable not set")
            return False        
        agent = DocumentChatAgent(
            openai_api_key=OPENAI_API_KEY,
            documents_folder="uploads",
            chunk_size=1000,
            chunk_overlap=200,
            temperature=0.7
        )
        logger.info("Agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return False

# Initialize agent on startup
initialize_agent()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload PDF file"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save file to uploads directory
        file_path = Path("uploads") / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process documents if agent is available
        if agent:
            success = agent.process_documents()
            if success:
                return {"message": f"File {file.filename} uploaded and processed successfully", "status": "success"}
            else:
                return {"message": "File uploaded but processing failed", "status": "warning"}
        else:
            return {"message": "File uploaded but agent not available", "status": "warning"}
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(question: str = Form(...)):
    """Chat endpoint"""
    try:
        if not agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        # Get response from agent
        result = agent.ask_question(question)
        
        if result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    """Get chat history"""
    try:
        if not agent:
            return {"history": []}
        
        history = agent.get_chat_history()
        return {"history": history}
        
    except Exception as e:
        logger.error(f"History error: {e}")
        return {"history": []}

@app.post("/clear")
async def clear_history():
    """Clear chat history"""
    try:
        if agent:
            agent.clear_memory()
        return {"message": "Chat history cleared", "status": "success"}
        
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def get_files():
    """Get list of uploaded files"""
    try:
        upload_dir = Path("uploads")
        if not upload_dir.exists():
            return {"files": []}
        
        files = [f.name for f in upload_dir.glob("*.pdf")]
        return {"files": files}
        
    except Exception as e:
        logger.error(f"Files error: {e}")
        return {"files": []}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 