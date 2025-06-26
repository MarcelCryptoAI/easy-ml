#!/usr/bin/env python3
import os
import uvicorn

if __name__ == "__main__":
    # Get port from Railway environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print(f"Starting server on port {port}")
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )