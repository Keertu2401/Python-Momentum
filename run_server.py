#!/usr/bin/env python3
"""
Enhanced Momentum Strategy API Server
Run this script to start the optimized momentum strategy API
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    """Start the API server with optimized settings"""
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if required files exist
    required_files = ['main.py', 'config.yaml']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all required files are present.")
        sys.exit(1)
    
    print("🚀 Starting Enhanced Momentum Strategy API v2.0...")
    print("📊 Features: Bug fixes, optimized performance, enhanced filtering")
    print("🔗 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    print("❤️  Health check: http://localhost:8000/health")
    print("\n" + "="*60)
    
    try:
        # Start the server with optimized settings
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,              # Auto-reload on code changes
            workers=1,                # Single worker for development
            access_log=True,          # Enable access logging
            log_level="info",         # Detailed logging
            reload_dirs=["./"],       # Watch current directory
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()