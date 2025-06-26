#!/usr/bin/env python3
import os
import sys
import uvicorn

# Add parent directory to Python path so we can import backend
sys.path.insert(0, '/app')

if __name__ == "__main__":
    # Get port from Railway environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print(f"Starting server on port {port}")
    print(f"Python path: {sys.path}")
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )