#!/usr/bin/env python3
"""
Launch script for the FastAPI application
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    from src.api.main import app
    
    print("Starting API server on http://0.0.0.0:8000")
    print("Swagger docs available at http://localhost:8000/docs")
    print("ReDoc available at http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
