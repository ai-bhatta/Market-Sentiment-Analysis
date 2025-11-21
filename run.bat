@echo off
REM Run script for Investor Market Sentiment Index (Windows)

echo Investor Market Sentiment Index - Quick Start
echo ==============================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run based on argument
if "%1"=="analyze" (
    echo Running sentiment analysis...
    python main.py
) else if "%1"=="quick" (
    echo Running quick test (100 articles^)...
    python main.py --quick
) else if "%1"=="dashboard" (
    echo Starting dashboard...
    streamlit run dashboard.py
) else if "%1"=="api" (
    echo Starting API server...
    python -m uvicorn src.api:app --reload
) else if "%1"=="docker" (
    echo Starting with Docker...
    docker-compose up --build
) else (
    echo Usage: run.bat [analyze^|quick^|dashboard^|api^|docker]
    echo.
    echo Options:
    echo   analyze    - Run full sentiment analysis
    echo   quick      - Run quick test with 100 articles
    echo   dashboard  - Start Streamlit dashboard
    echo   api        - Start FastAPI server
    echo   docker     - Run with Docker Compose
)
