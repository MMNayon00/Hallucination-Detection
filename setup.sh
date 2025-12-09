#!/bin/bash

# Setup script for Hallucination Detection System
# For MacBook Air M2 / macOS

echo "========================================="
echo "Hallucination Detection System - Setup"
echo "========================================="

# Check Python version
echo ""
echo "[1/4] Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.10 or higher from https://www.python.org"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/4] Creating virtual environment..."
cd backend
python3 -m venv venv

# Activate virtual environment
echo ""
echo "[3/4] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[4/4] Installing dependencies..."
echo "This will download ~400MB of models. Please wait..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the server:"
echo "   python app.py"
echo ""
echo "3. Open frontend/index.html in your browser"
echo ""
echo "Or run evaluation:"
echo "   python run_evaluation.py"
echo ""
echo "========================================="
