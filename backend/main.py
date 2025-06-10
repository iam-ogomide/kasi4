from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from bot_service import BotService
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Kasi - CreditChek API Assistant",
    description="AI assistant for CreditChek API integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize BotService
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
bot_service = BotService(FASTAPI_URL)

# Request models
class ChatRequest(BaseModel):
    message: str
    language: str

class LanguageRequest(BaseModel):
    language: str

# API endpoints
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Processing chat request: {request.message}")
        bot_service.set_language_preference(request.language)
        response = bot_service.generate_response(request.message)
        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts")
async def get_prompts():
    try:
        prompts = bot_service.get_preconfigured_prompts()
        return prompts
    except Exception as e:
        logger.error(f"Error getting prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/prompts")
async def get_prompts():
    try:
        prompts = bot_service.get_preconfigured_prompts()
        return prompts
    except Exception as e:
        logger.error(f"Error getting prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/language")
async def set_language(request: LanguageRequest):
    try:
        bot_service.set_language_preference(request.language)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error setting language: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend files in production
if os.path.exists("../frontend"):
    app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)