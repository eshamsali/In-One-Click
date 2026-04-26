#!/bin/bash
echo "========================================"
echo "  StatLab — Starting Backend Server"
echo "========================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Download from: https://www.python.org/downloads/"
    exit 1
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip3 install -r requirements.txt -q

# Start Flask server
echo ""
echo "Server starting at: http://localhost:5000"
echo "Open your browser and go to:  index.html"
echo "Press Ctrl+C to stop the server"
echo "========================================"
cd "$(dirname "$0")"
python3 analysis.py
