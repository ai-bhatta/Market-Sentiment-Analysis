import pandas as pd
from pathlib import Path
from datetime import datetime
import re


class NewsDataLoader:
    """Load and preprocess financial news datasets"""

    def __init__(self, dataset_path="dataset"):
        self.dataset_path = Path(dataset_path)

    def load_cnbc_data(self):
        """Load CNBC headlines"""
        file_path = self.dataset_path / "cnbc_headlines.csv"
        df = pd.read_csv(file_path)

        # Clean the data
        df = df.dropna(subset=['Headlines'])

        # Extract time information
        df['Time'] = df['Time'].fillna('')

        # Create a text column combining headline and description
        df['text'] = df['Headlines'].astype(str)
        if 'Description' in df.columns:
            df['description'] = df['Description'].fillna('')
            df['text'] = df['text'] + '. ' + df['description']

        df['source'] = 'CNBC'

        return df[['text', 'Time', 'source']].copy()

    def load_guardian_data(self):
        """Load Guardian headlines"""
        file_path = self.dataset_path / "guardian_headlines.csv"
        df = pd.read_csv(file_path)

        # Clean the data
        df = df.dropna(subset=['Headlines'])

        # Rename columns for consistency
        df['text'] = df['Headlines'].astype(str)
        df['source'] = 'Guardian'

        return df[['text', 'Time', 'source']].copy()

    def load_reuters_data(self):
        """Load Reuters headlines"""
        file_path = self.dataset_path / "reuters_headlines.csv"
        df = pd.read_csv(file_path)

        # Clean the data
        df = df.dropna(subset=['Headlines'])

        # Create text column
        df['text'] = df['Headlines'].astype(str)
        if 'Description' in df.columns:
            df['description'] = df['Description'].fillna('')
            df['text'] = df['text'] + '. ' + df['description']

        df['source'] = 'Reuters'

        return df[['text', 'Time', 'source']].copy()

    def parse_date(self, date_str):
        """Parse various date formats"""
        if pd.isna(date_str) or date_str == '':
            return None

        # Try different date formats
        formats = [
            '%d-%b-%y',
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%B %d, %Y'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(str(date_str).split(',')[0].strip(), fmt)
            except:
                continue

        # If all fail, try to extract date from string
        try:
            # Look for patterns like "Jul 18 2020"
            match = re.search(r'(\w+)\s+(\d+)\s+(\d+)', str(date_str))
            if match:
                return datetime.strptime(f"{match.group(1)} {match.group(2)} {match.group(3)}", '%b %d %Y')
        except:
            pass

        return None

    def load_all_data(self):
        """Load and combine all news sources"""
        print("Loading CNBC data...")
        cnbc_df = self.load_cnbc_data()

        print("Loading Guardian data...")
        guardian_df = self.load_guardian_data()

        print("Loading Reuters data...")
        reuters_df = self.load_reuters_data()

        # Combine all datasets
        combined_df = pd.concat([cnbc_df, guardian_df, reuters_df], ignore_index=True)

        # Parse dates
        combined_df['date'] = combined_df['Time'].apply(self.parse_date)

        # Clean text
        combined_df['text'] = combined_df['text'].apply(self.clean_text)

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text'])

        # Sort by date
        combined_df = combined_df.sort_values('date', ascending=False)

        print(f"Loaded {len(combined_df)} total news articles")
        print(f"CNBC: {len(cnbc_df)}, Guardian: {len(guardian_df)}, Reuters: {len(reuters_df)}")

        return combined_df

    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', str(text))

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\$\%]', '', text)

        return text.strip()


if __name__ == "__main__":
    # Test the data loader
    loader = NewsDataLoader()
    df = loader.load_all_data()
    print("\nSample data:")
    print(df.head())
    print("\nData info:")
    print(df.info())
