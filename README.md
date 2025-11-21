# Investor Market Sentiment Index using FinBERT

A real-time market sentiment analysis tool that fetches live financial news and processes it using FinBERT to generate an aggregated sentiment index. The system provides both historical analysis of news datasets and live sentiment tracking with an interactive dashboard.

## Features

- **Live News Fetching**: Real-time news from multiple financial sources (NewsAPI, NewsData.io, and RSS feeds)
- **Historical Analysis**: Process pre-collected news from CNBC, Reuters, and The Guardian
- **FinBERT Integration**: Uses state-of-the-art financial sentiment analysis model
- **Sentiment Index**: Aggregates individual sentiments into a comprehensive market index (-100 to +100 scale)
- **Interactive Live Dashboard**: Real-time visualization with auto-refresh using Streamlit
- **REST API**: FastAPI endpoints for programmatic access
- **Historical Tracking**: SQLite database for sentiment trends over time
- **Financial Text Validation**: Filters non-financial content to ensure accurate sentiment
- **Multiple Visualization Types**: Line charts, bar charts, pie charts, and tables

## Current Status

**Version**: 1.0 (Fully Functional)

The project is currently operational with the following components:
- ✅ Live news fetching from multiple sources
- ✅ FinBERT sentiment analysis pipeline
- ✅ Real-time dashboard with auto-refresh
- ✅ SQLite database for historical data
- ✅ RESTful API endpoints
- ✅ Docker support for containerized deployment
- ✅ Cross-platform scripts (Windows batch & Unix shell)

## Project Structure

```
Project_MSA/
├── dataset/                    # Historical news data (11.8 MB total)
│   ├── cnbc_headlines.csv      # 666 KB
│   ├── guardian_headlines.csv  # 1.4 MB
│   └── reuters_headlines.csv   # 9.5 MB
├── src/
│   ├── __init__.py
│   ├── data_loader.py          
│   ├── sentiment_analyzer.py  
│   ├── sentiment_index.py      
│   ├── database.py             
│   ├── news_fetcher.py         
│   └── api.py                  
├── dashboard_live.py           
├── .env.example                
├── requirements.txt            
├── Dockerfile                  
├── docker-compose.yml          
├── run.sh                      
├── run.bat                     
└── sentiment_data.db           
```

## How to Run This Project Locally

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package installer
- **Git**: To clone the repository (optional)
- **Disk Space**: ~2 GB for models and dependencies
- **RAM**: Minimum 4 GB (8 GB recommended for faster processing)

### Option 1: Quick Start with Scripts (Recommended)

#### On Windows:
```cmd
# Run the live dashboard
run.bat dashboard

# Run sentiment analysis on historical data
run.bat analyze

# Start the API server
run.bat api
```

#### On Linux/Mac:
```bash
# Make the script executable (first time only)
chmod +x run.sh

# Run the live dashboard
./run.sh dashboard

# Run sentiment analysis on historical data
./run.sh analyze

# Start the API server
./run.sh api
```

### Option 2: Manual Setup

#### Step 1: Clone or Download the Project
```bash
git clone <repository-url>
cd Project_MSA
```

#### Step 2: Create a Virtual Environment
**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- Transformers (Hugging Face library for FinBERT)
- Streamlit (dashboard framework)
- FastAPI & Uvicorn (API server)
- Pandas, NumPy (data processing)
- Plotly (interactive visualizations)
- SQLAlchemy (database ORM)
- Requests, Feedparser (news fetching)

**Note**: First-time installation will download the FinBERT model (~450 MB) automatically when you first run the application.

#### Step 4: Configure Environment Variables (Optional)
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your preferred settings (optional)
# The application works with defaults if .env is not present
```

#### Step 5: Run the Application

**Option A - Live Dashboard (Recommended for First-Time Users)**
```bash
streamlit run dashboard_live.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`

**Option B - API Server**
```bash
python -m uvicorn src.api:app --reload
```
Access the API at `http://localhost:8000` and docs at `http://localhost:8000/docs`

**Option C - Historical Analysis**
```bash
python main.py
```

### Option 3: Docker Deployment

If you have Docker and Docker Compose installed:

```bash
# Build and start all services
docker-compose up --build

# Or use the run scripts
./run.sh docker     # Linux/Mac
run.bat docker      # Windows
```

This will start both the dashboard and API server in containers.

## Usage Guide

### Live Dashboard Features

1. **Auto-Refresh**: Dashboard updates every 5 minutes with latest news
2. **Manual Refresh**: Click "Fetch Latest News Now" button
3. **Custom Text Analysis**: Enter your own text in the sidebar
4. **Historical View**: View sentiment trends over time
5. **Source Breakdown**: See sentiment by news source
6. **Recent Headlines**: Browse analyzed articles with sentiment scores

### API Endpoints

Once the API server is running, access these endpoints:

- `GET /sentiment/latest` - Get the latest sentiment index
- `GET /sentiment/history` - Retrieve historical sentiment data
- `GET /sentiment/by-source` - Get sentiment breakdown by news source
- `POST /sentiment/analyze` - Analyze custom text

**Example API Usage:**
```bash
# Get latest sentiment
curl http://localhost:8000/sentiment/latest

# Analyze custom text
curl -X POST http://localhost:8000/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Stock markets hit record highs today"}'
```

Interactive API documentation is available at `http://localhost:8000/docs`

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'transformers'`
- **Solution**: Activate your virtual environment and run `pip install -r requirements.txt`

**Issue**: CUDA/GPU errors
- **Solution**: The application will automatically fall back to CPU. If you want GPU support, install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Issue**: Dashboard not loading
- **Solution**: Ensure port 8501 is not in use. Change the port with: `streamlit run dashboard_live.py --server.port 8502`

**Issue**: API server fails to start
- **Solution**: Port 8000 might be in use. Change it with: `uvicorn src.api:app --port 8001`

**Issue**: Slow performance
- **Solution**: Reduce batch size in `.env` (set `BATCH_SIZE=8`) or limit articles analyzed

**Issue**: News fetching fails
- **Solution**: Check your internet connection. The app uses free news APIs that may have rate limits.

### System Requirements

- **Minimum**: Python 3.8, 4GB RAM, 2GB free disk space
- **Recommended**: Python 3.10+, 8GB RAM, 5GB free disk space, GPU (optional)

## Model Information

This project uses [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model pre-trained on financial communication text and fine-tuned for sentiment analysis. The model classifies text into three categories:
- **Positive**: Optimistic financial outlook
- **Negative**: Pessimistic financial outlook
- **Neutral**: Factual or balanced information

The sentiment index is calculated as: `(Positive - Negative) * 100`, ranging from -100 (extremely bearish) to +100 (extremely bullish).

## Contributing

This is an academic project for Year 4 university coursework. Contributions, suggestions, and feedback are welcome.

## License

MIT License
