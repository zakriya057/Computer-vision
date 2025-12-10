from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Computer Vision API",
    description="A FastAPI application for computer vision tasks",
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


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Computer Vision API"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
