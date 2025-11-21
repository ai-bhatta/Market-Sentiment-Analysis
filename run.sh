#!/bin/bash

# Run script for Investor Market Sentiment Index

echo "Investor Market Sentiment Index - Quick Start"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run based on argument
case "$1" in
    "analyze")
        echo "Running sentiment analysis..."
        python main.py
        ;;
    "quick")
        echo "Running quick test (100 articles)..."
        python main.py --quick
        ;;
    "dashboard")
        echo "Starting dashboard..."
        streamlit run dashboard.py
        ;;
    "api")
        echo "Starting API server..."
        python -m uvicorn src.api:app --reload
        ;;
    "docker")
        echo "Starting with Docker..."
        docker-compose up --build
        ;;
    *)
        echo "Usage: ./run.sh [analyze|quick|dashboard|api|docker]"
        echo ""
        echo "Options:"
        echo "  analyze    - Run full sentiment analysis"
        echo "  quick      - Run quick test with 100 articles"
        echo "  dashboard  - Start Streamlit dashboard"
        echo "  api        - Start FastAPI server"
        echo "  docker     - Run with Docker Compose"
        ;;
esac
