#!/bin/bash

# Momentum Strategy API Startup Script

echo "🚀 Starting Momentum Strategy API..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "config.yaml" ]; then
    echo "❌ config.yaml not found. Please create it first."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Start the API
echo "🌟 Starting API server on http://localhost:8000"
echo "📖 API documentation available at http://localhost:8000/docs"
echo "🔍 Health check at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 main.py