# Add this to the END of your main.py file to fix deployment issues

# Remove or comment out any existing "if __name__ == '__main__':" block at the end
# Replace it with this deployment-ready version:

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (required for Railway/Render)
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server with deployment-friendly settings
    uvicorn.run(
        "main:app",  # Reference to your FastAPI app
        host="0.0.0.0",  # Listen on all interfaces
        port=port,  # Use environment port
        reload=False,  # Disable reload in production
        access_log=True,  # Enable access logging
        log_level="info"  # Set appropriate log level
    )