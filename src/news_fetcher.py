"""
Live Financial News Fetcher
Fetches real-time financial news from multiple sources using free APIs
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import time
import sys

# Fix Windows console encoding for emojis
def safe_print(text):
    """Print text safely, handling encoding issues on Windows"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Remove emojis and special characters for Windows console
        clean_text = text.encode('ascii', 'ignore').decode('ascii')
        print(clean_text)


class FinancialNewsFetcher:
    """Fetch live financial news from multiple sources"""

    def __init__(self):
        # Financial keywords for filtering
        self.financial_keywords = [
            'stock', 'market', 'trading', 'investor', 'economy', 'financial',
            'earnings', 'revenue', 'profit', 'loss', 'shares', 'equity',
            'bond', 'fed', 'interest rate', 'inflation', 'gdp', 'dollar',
            'currency', 'forex', 'commodity', 'oil', 'gold', 'crypto',
            'bitcoin', 'ethereum', 'nasdaq', 'dow jones', 's&p', 'nyse',
            'merger', 'acquisition', 'ipo', 'dividend', 'quarterly',
            'fiscal', 'monetary', 'central bank', 'treasury', 'debt',
            'credit', 'loan', 'mortgage', 'investment', 'fund', 'etf',
            'hedge fund', 'portfolio', 'asset', 'valuation', 'pe ratio',
            'bull market', 'bear market', 'recession', 'recovery', 'growth'
        ]

    def is_financial_content(self, text: str) -> bool:
        """
        Check if text is finance-related
        Returns True if text contains financial keywords
        """
        if not text:
            return False

        text_lower = text.lower()

        # Check for financial keywords
        keyword_count = sum(1 for keyword in self.financial_keywords
                          if keyword in text_lower)

        # Require at least 2 financial keywords for validation
        return keyword_count >= 2

    def fetch_newsapi_org(self, api_key: str = None, days: int = 7) -> pd.DataFrame:
        """
        Fetch financial news from NewsAPI.org
        Get free API key at: https://newsapi.org/register
        """
        if not api_key:
            safe_print("[INFO] NewsAPI key not provided. Skipping NewsAPI.org...")
            return pd.DataFrame()

        url = "https://newsapi.org/v2/everything"

        # Financial keywords for query
        query = "(stock OR market OR trading OR earnings OR economy OR financial OR investment)"

        # Date range
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,
            'apiKey': api_key
        }

        try:
            safe_print(f"[FETCH] Fetching news from NewsAPI.org (last {days} days)...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            articles = data.get('articles', [])

            if not articles:
                safe_print("No articles found from NewsAPI.org")
                return pd.DataFrame()

            # Process articles
            processed = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title}. {description}"

                # Validate financial content
                if self.is_financial_content(content):
                    processed.append({
                        'text': content,
                        'Time': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                        'url': article.get('url', '')
                    })

            df = pd.DataFrame(processed)
            safe_print(f"✅ Fetched {len(df)} financial articles from NewsAPI.org")
            return df

        except Exception as e:
            safe_print(f"❌ Error fetching from NewsAPI.org: {e}")
            return pd.DataFrame()

    def fetch_guardian(self, api_key: str = None, days: int = 7) -> pd.DataFrame:
        """
        Fetch financial news from The Guardian API
        Get free API key at: https://open-platform.theguardian.com/access/
        """
        if not api_key:
            safe_print("⚠️  Guardian API key not provided. Skipping Guardian...")
            return pd.DataFrame()

        url = "https://content.guardianapis.com/search"

        # Date range
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        params = {
            'section': 'business',
            'from-date': from_date,
            'page-size': 50,
            'show-fields': 'headline,bodyText,trailText',
            'api-key': api_key
        }

        try:
            safe_print(f"📰 Fetching news from The Guardian (last {days} days)...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            articles = data.get('response', {}).get('results', [])

            if not articles:
                safe_print("No articles found from The Guardian")
                return pd.DataFrame()

            # Process articles
            processed = []
            for article in articles:
                fields = article.get('fields', {})
                headline = fields.get('headline', '')
                trail_text = fields.get('trailText', '')
                content = f"{headline}. {trail_text}"

                # Validate financial content
                if self.is_financial_content(content):
                    processed.append({
                        'text': content,
                        'Time': article.get('webPublicationDate', ''),
                        'source': 'The Guardian',
                        'url': article.get('webUrl', '')
                    })

            df = pd.DataFrame(processed)
            safe_print(f"✅ Fetched {len(df)} financial articles from The Guardian")
            return df

        except Exception as e:
            safe_print(f"❌ Error fetching from The Guardian: {e}")
            return pd.DataFrame()

    def fetch_finnhub(self, api_key: str = None) -> pd.DataFrame:
        """
        Fetch market news from Finnhub (Financial data API)
        Get free API key at: https://finnhub.io/register
        """
        if not api_key:
            safe_print("⚠️  Finnhub API key not provided. Skipping Finnhub...")
            return pd.DataFrame()

        url = "https://finnhub.io/api/v1/news"

        params = {
            'category': 'general',
            'token': api_key
        }

        try:
            safe_print("📰 Fetching news from Finnhub...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            articles = response.json()

            if not articles:
                safe_print("No articles found from Finnhub")
                return pd.DataFrame()

            # Process articles
            processed = []
            for article in articles:
                headline = article.get('headline', '')
                summary = article.get('summary', '')
                content = f"{headline}. {summary}"

                # All Finnhub content is financial
                processed.append({
                    'text': content,
                    'Time': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                    'source': article.get('source', 'Finnhub'),
                    'url': article.get('url', '')
                })

            df = pd.DataFrame(processed)
            safe_print(f"✅ Fetched {len(df)} financial articles from Finnhub")
            return df

        except Exception as e:
            safe_print(f"❌ Error fetching from Finnhub: {e}")
            return pd.DataFrame()

    def fetch_alpha_vantage_news(self, api_key: str = None) -> pd.DataFrame:
        """
        Fetch market news from Alpha Vantage
        Get free API key at: https://www.alphavantage.co/support/#api-key
        """
        if not api_key:
            safe_print("⚠️  Alpha Vantage API key not provided. Skipping...")
            return pd.DataFrame()

        url = "https://www.alphavantage.co/query"

        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': api_key
        }

        try:
            safe_print("📰 Fetching news from Alpha Vantage...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            articles = data.get('feed', [])

            if not articles:
                safe_print("No articles found from Alpha Vantage")
                return pd.DataFrame()

            # Process articles
            processed = []
            for article in articles:
                title = article.get('title', '')
                summary = article.get('summary', '')
                content = f"{title}. {summary}"

                processed.append({
                    'text': content,
                    'Time': article.get('time_published', ''),
                    'source': article.get('source', 'Alpha Vantage'),
                    'url': article.get('url', '')
                })

            df = pd.DataFrame(processed)
            safe_print(f"✅ Fetched {len(df)} financial articles from Alpha Vantage")
            return df

        except Exception as e:
            safe_print(f"❌ Error fetching from Alpha Vantage: {e}")
            return pd.DataFrame()

    def fetch_all(self, api_keys: Dict[str, str] = None, days: int = 7) -> pd.DataFrame:
        """
        Fetch from all available sources

        Args:
            api_keys: Dictionary with keys:
                - 'newsapi': NewsAPI.org key
                - 'guardian': The Guardian key
                - 'finnhub': Finnhub key
                - 'alphavantage': Alpha Vantage key
            days: Number of days to fetch

        Returns:
            Combined DataFrame with all articles
        """
        if api_keys is None:
            api_keys = {}

        safe_print("\n" + "="*80)
        safe_print("🔍 FETCHING LIVE FINANCIAL NEWS")
        safe_print("="*80 + "\n")

        all_articles = []

        # Fetch from each source
        sources = [
            ('newsapi', self.fetch_newsapi_org, {'api_key': api_keys.get('newsapi'), 'days': days}),
            ('guardian', self.fetch_guardian, {'api_key': api_keys.get('guardian'), 'days': days}),
            ('finnhub', self.fetch_finnhub, {'api_key': api_keys.get('finnhub')}),
            ('alphavantage', self.fetch_alpha_vantage_news, {'api_key': api_keys.get('alphavantage')}),
        ]

        for source_name, fetch_func, kwargs in sources:
            try:
                df = fetch_func(**kwargs)
                if not df.empty:
                    all_articles.append(df)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                safe_print(f"❌ Error with {source_name}: {e}")

        # Combine all articles
        if not all_articles:
            safe_print("\n⚠️  No articles fetched. Using fallback RSS feeds...\n")
            return self.fetch_rss_fallback()

        combined = pd.concat(all_articles, ignore_index=True)

        # Remove duplicates
        combined = combined.drop_duplicates(subset=['text'], keep='first')

        # Parse dates
        combined['date'] = pd.to_datetime(combined['Time'], errors='coerce')

        # Sort by date
        combined = combined.sort_values('date', ascending=False)

        safe_print("\n" + "="*80)
        print(f"✅ TOTAL ARTICLES FETCHED: {len(combined)}")
        safe_print("="*80 + "\n")

        return combined

    def fetch_rss_fallback(self) -> pd.DataFrame:
        """
        Fallback method using RSS feeds (no API key required)
        """
        import feedparser

        safe_print("📰 Using RSS feeds (no API keys required)...\n")

        rss_feeds = {
            'Reuters Business': 'http://feeds.reuters.com/reuters/businessNews',
            'CNBC': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'MarketWatch': 'http://feeds.marketwatch.com/marketwatch/topstories',
            'Bloomberg': 'https://www.bloomberg.com/feed/podcast/markets-daily.xml',
        }

        all_articles = []

        for source_name, feed_url in rss_feeds.items():
            try:
                safe_print(f"📰 Fetching from {source_name}...")
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:20]:  # Limit to 20 per source
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    content = f"{title}. {summary}"

                    # Validate financial content
                    if self.is_financial_content(content):
                        all_articles.append({
                            'text': content,
                            'Time': entry.get('published', datetime.now().isoformat()),
                            'source': source_name,
                            'url': entry.get('link', '')
                        })

                safe_print(f"✅ Fetched {len([a for a in all_articles if a['source'] == source_name])} articles")
                time.sleep(1)

            except Exception as e:
                safe_print(f"❌ Error with {source_name}: {e}")

        if not all_articles:
            safe_print("⚠️  No articles from RSS feeds either!")
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        df['date'] = pd.to_datetime(df['Time'], errors='coerce')

        print(f"\n✅ Total from RSS: {len(df)} articles\n")

        return df


def validate_financial_text(text: str) -> tuple[bool, str]:
    """
    Validate if text is financial content and return reason

    Returns:
        (is_valid, reason)
    """
    fetcher = FinancialNewsFetcher()

    if not text or len(text.strip()) < 10:
        return False, "Text too short (minimum 10 characters)"

    if not fetcher.is_financial_content(text):
        return False, "Text does not contain sufficient financial keywords"

    return True, "Valid financial content"


if __name__ == "__main__":
    # Example usage
    fetcher = FinancialNewsFetcher()

    # Test financial validation
    test_texts = [
        "Apple stock surges on strong earnings report",
        "I love pizza and ice cream",
        "The Federal Reserve raised interest rates to combat inflation",
        "My cat is very cute today"
    ]

    safe_print("Testing financial content validation:\n")
    for text in test_texts:
        is_financial = fetcher.is_financial_content(text)
        print(f"{'✅' if is_financial else '❌'} {text[:50]}...")

    safe_print("\n" + "="*80)
    safe_print("To fetch live news, use:")
    safe_print("="*80)
    safe_print("""
api_keys = {
    'newsapi': 'your_newsapi_key',
    'guardian': 'your_guardian_key',
    'finnhub': 'your_finnhub_key',
    'alphavantage': 'your_alphavantage_key'
}

df = fetcher.fetch_all(api_keys=api_keys, days=7)
print(df.head())
""")
