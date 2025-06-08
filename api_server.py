from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from ai_mechanic_core_enhanced import EnhancedAIMechanicCore

# Initialize FastAPI app
app = FastAPI(
    title="Radical Assistant API",
    description="AI Assistant for Radical SR3 Owner's Manual and Handling Guide",
    version="2.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trackmeai-frontend.vercel.app",  # New frontend domain
        "https://trackitai.vercel.app",  # Keep old domain for transition
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Enhanced AI Mechanic
try:
    ai_mechanic = EnhancedAIMechanicCore()
    print("✓ Radical Assistant initialized successfully")
    print("✓ PDF documentation search system ready")
except Exception as e:
    print(f"✗ Failed to initialize Radical Assistant: {e}")
    ai_mechanic = None

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class SourceChunk(BaseModel):
    content: str
    source_type: str  # 'trackmeai_docs', 'internet_search', 'general_ai'
    confidence: float
    metadata: Dict
    url: Optional[str] = None

class EnhancedAIMechanicResponse(BaseModel):
    answer: str
    source_chunks: List[SourceChunk]
    question: str
    top_k: int
    total_confidence: float
    has_trackmeai_sources: bool
    has_internet_sources: bool
    error: Optional[bool] = False

# Keep backward compatibility
AIMechanicResponse = EnhancedAIMechanicResponse

class HealthResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Radical Assistant API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "ask": "/ask-ai-mechanic",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if ai_mechanic is None:
        return HealthResponse(
            status="error",
            message="Radical Assistant not initialized",
            documents_loaded=0
        )
    
    return HealthResponse(
        status="healthy",
        message="Radical Assistant is running",
        documents_loaded=len(ai_mechanic.documents)
    )

@app.post("/ask-ai-mechanic", response_model=AIMechanicResponse)
async def ask_ai_mechanic(request: QuestionRequest):
    """
    Ask the Radical Assistant a question about the Radical SR3 manuals
    
    This endpoint performs:
    1. Vector similarity search on manual chunks
    2. Context building with top matching sections
    3. GPT-4o query with the context
    4. Returns answer with source chunks
    """
    
    if ai_mechanic is None:
        raise HTTPException(
            status_code=503, 
            detail="Radical Assistant service not available"
        )
    
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    # Validate top_k parameter
    if request.top_k < 1 or request.top_k > 10:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 10"
        )
    
    try:
        # Process the question using AI Mechanic
        response = ai_mechanic.ask_ai_mechanic(
            question=request.question.strip(),
            top_k=request.top_k
        )
        
        # Debug: Print response structure
        print(f"DEBUG: Response type: {type(response)}")
        print(f"DEBUG: Response keys: {list(response.keys())}")
        if response["source_chunks"]:
            print(f"DEBUG: First source chunk keys: {list(response['source_chunks'][0].keys())}")
        
        # Convert to API response format
        source_chunks = [
            SourceChunk(
                content=chunk["content"],
                source_type=chunk["source_type"],
                confidence=chunk["confidence"],
                metadata=chunk["metadata"],
                url=chunk.get("url")
            )
            for chunk in response["source_chunks"]
        ]
        
        return AIMechanicResponse(
            answer=response["answer"],
            source_chunks=source_chunks,
            question=response["question"],
            top_k=response["top_k"],
            total_confidence=response["total_confidence"],
            has_trackmeai_sources=response["has_trackmeai_sources"],
            has_internet_sources=response["has_internet_sources"],
            error=response.get("error", False)
        )
        
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error response
        return AIMechanicResponse(
            answer=f"Sorry, I encountered an error while processing your question: {str(e)}",
            source_chunks=[],
            question=request.question,
            top_k=request.top_k,
            total_confidence=0.0,
            has_trackmeai_sources=False,
            has_internet_sources=False,
            error=True
        )

@app.get("/manuals", response_model=List[str])
async def get_available_manuals():
    """Get list of available manuals"""
    if ai_mechanic is None:
        raise HTTPException(
            status_code=503,
            detail="Radical Assistant service not available"
        )
    
    # Get unique manual IDs
    manual_ids = list(set(doc["manual_id"] for doc in ai_mechanic.documents))
    return sorted(manual_ids)

@app.get("/sections/{manual_id}", response_model=List[Dict[str, str]])
async def get_manual_sections(manual_id: str):
    """Get all sections for a specific manual"""
    if ai_mechanic is None:
        raise HTTPException(
            status_code=503,
            detail="Radical Assistant service not available"
        )
    
    # Filter documents by manual_id
    sections = [
        {
            "section_title": doc["section_title"],
            "page_start": str(doc.get("page_start", "Unknown"))
        }
        for doc in ai_mechanic.documents
        if doc["manual_id"] == manual_id
    ]
    
    if not sections:
        raise HTTPException(
            status_code=404,
            detail=f"Manual '{manual_id}' not found"
        )
    
    return sections

# Custom exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": str(exc),
        "status_code": 500
    }

def main():
    """Run the FastAPI server"""
    print("Starting Radical Assistant API server...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("API endpoints:")
    print("  GET  /health - Health check")
    print("  POST /ask-ai-mechanic - Ask questions")
    print("  GET  /manuals - List available manuals")
    print("  GET  /sections/{manual_id} - Get manual sections")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()