from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import pandas as pd

from .database import SentimentDatabase
from .sentiment_analyzer import FinBERTSentimentAnalyzer
from .data_loader import NewsDataLoader
from .sentiment_index import SentimentIndexCalculator

app = FastAPI(
    title="Investor Market Sentiment Index API",
    description="API for analyzing financial news sentiment using FinBERT",
    version="1.0.0"
)

# Initialize components
db = SentimentDatabase()
analyzer = None  # Lazy load to avoid loading model at startup
calculator = SentimentIndexCalculator()


def get_analyzer():
    """Lazy load the sentiment analyzer"""
    global analyzer
    if analyzer is None:
        analyzer = FinBERTSentimentAnalyzer()
    return analyzer


class TextInput(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    score: float
    positive: float
    negative: float
    neutral: float
    compound: float


class IndexResponse(BaseModel):
    date: str
    sentiment_index: float
    interpretation: str
    avg_compound: float
    volatility: float
    momentum: Optional[float]
    article_count: int


@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "Investor Market Sentiment Index API",
        "version": "1.0.0",
        "endpoints": {
            "sentiment_latest": "/sentiment/latest",
            "sentiment_history": "/sentiment/history",
            "sentiment_by_source": "/sentiment/by-source",
            "analyze_text": "/sentiment/analyze",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/sentiment/latest", response_model=IndexResponse)
def get_latest_sentiment():
    """Get the latest sentiment index"""
    latest = db.get_latest_index()

    if not latest:
        raise HTTPException(status_code=404, detail="No sentiment data available")

    interpretation = calculator.interpret_index(latest['sentiment_index'])

    return {
        "date": latest['date'],
        "sentiment_index": latest['sentiment_index'],
        "interpretation": interpretation,
        "avg_compound": latest['avg_compound'],
        "volatility": latest['volatility'],
        "momentum": latest['momentum'],
        "article_count": latest['article_count']
    }


@app.get("/sentiment/history")
def get_sentiment_history(days: int = 30):
    """
    Get historical sentiment index data

    Args:
        days: Number of days of history to retrieve (default: 30)
    """
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365")

    history = db.get_historical_index(days=days)

    if not history:
        raise HTTPException(status_code=404, detail="No historical data available")

    # Add interpretation to each record
    for record in history:
        record['interpretation'] = calculator.interpret_index(record['sentiment_index'])

    return {
        "period_days": days,
        "data_points": len(history),
        "history": history
    }


@app.get("/sentiment/by-source")
def get_sentiment_by_source(days: int = 7):
    """
    Get sentiment breakdown by news source

    Args:
        days: Number of days to analyze (default: 7)
    """
    if days < 1 or days > 90:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 90")

    by_source = db.get_sentiment_by_source(days=days)

    if not by_source:
        raise HTTPException(status_code=404, detail="No source data available")

    # Add interpretation
    for record in by_source:
        record['interpretation'] = calculator.interpret_index(record['sentiment_index'])

    return {
        "period_days": days,
        "sources": by_source
    }


@app.post("/sentiment/analyze", response_model=SentimentResponse)
def analyze_text(input_data: TextInput):
    """
    Analyze sentiment of custom text

    Args:
        input_data: Text to analyze
    """
    if not input_data.text or len(input_data.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(input_data.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")

    # Get analyzer
    sentiment_analyzer = get_analyzer()

    # Analyze sentiment
    result = sentiment_analyzer.analyze_sentiment(input_data.text)

    return {
        "label": result['label'],
        "score": result['score'],
        "positive": result['positive'],
        "negative": result['negative'],
        "neutral": result['neutral'],
        "compound": result['positive'] - result['negative']
    }


@app.get("/sentiment/statistics")
def get_statistics():
    """Get overall sentiment statistics"""
    latest = db.get_latest_index()
    history_30 = db.get_historical_index(days=30)

    if not latest:
        raise HTTPException(status_code=404, detail="No sentiment data available")

    # Calculate statistics from history
    if history_30:
        indices = [h['sentiment_index'] for h in history_30]
        import statistics

        stats = {
            "current_index": latest['sentiment_index'],
            "current_interpretation": calculator.interpret_index(latest['sentiment_index']),
            "30_day_avg": statistics.mean(indices),
            "30_day_high": max(indices),
            "30_day_low": min(indices),
            "30_day_volatility": statistics.stdev(indices) if len(indices) > 1 else 0,
            "total_articles": latest['article_count']
        }
    else:
        stats = {
            "current_index": latest['sentiment_index'],
            "current_interpretation": calculator.interpret_index(latest['sentiment_index']),
            "total_articles": latest['article_count']
        }

    return stats


if __name__ == "__main__":
    import uvicorn
    print("Starting Sentiment Index API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
