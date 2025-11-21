import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from tqdm import tqdm


class FinBERTSentimentAnalyzer:
    """Sentiment analysis using FinBERT model"""

    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize FinBERT model and tokenizer

        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading FinBERT model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.labels = ['positive', 'negative', 'neutral']

        print("FinBERT model loaded successfully!")

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a single text

        Args:
            text: Input text string

        Returns:
            dict: Sentiment scores and label
        """
        if not text or pd.isna(text):
            return {
                'label': 'neutral',
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'score': 0.0
            }

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to numpy
        scores = predictions.cpu().numpy()[0]

        # Create result dictionary
        result = {
            'positive': float(scores[0]),
            'negative': float(scores[1]),
            'neutral': float(scores[2])
        }

        # Determine label and confidence
        max_idx = np.argmax(scores)
        result['label'] = self.labels[max_idx]
        result['score'] = float(scores[max_idx])

        return result

    def analyze_batch(self, texts, batch_size=32):
        """
        Analyze sentiment for multiple texts

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            list: List of sentiment dictionaries
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiments"):
            batch = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Process each prediction in batch
            scores_batch = predictions.cpu().numpy()

            for scores in scores_batch:
                result = {
                    'positive': float(scores[0]),
                    'negative': float(scores[1]),
                    'neutral': float(scores[2])
                }

                max_idx = np.argmax(scores)
                result['label'] = self.labels[max_idx]
                result['score'] = float(scores[max_idx])

                results.append(result)

        return results

    def analyze_dataframe(self, df, text_column='text', batch_size=32):
        """
        Analyze sentiment for a pandas DataFrame

        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            batch_size: Batch size for processing

        Returns:
            DataFrame: Original df with sentiment columns added
        """
        print(f"Analyzing {len(df)} texts...")

        texts = df[text_column].fillna('').tolist()
        sentiments = self.analyze_batch(texts, batch_size)

        # Add sentiment columns
        df = df.copy()
        df['sentiment_label'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        df['positive_score'] = [s['positive'] for s in sentiments]
        df['negative_score'] = [s['negative'] for s in sentiments]
        df['neutral_score'] = [s['neutral'] for s in sentiments]

        # Calculate compound sentiment score (-1 to 1)
        df['compound_score'] = df['positive_score'] - df['negative_score']

        print("Sentiment analysis complete!")
        print(f"\nSentiment distribution:")
        print(df['sentiment_label'].value_counts())

        return df


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = FinBERTSentimentAnalyzer()

    # Test single text
    test_text = "The stock market reached record highs as investors show confidence in economic recovery."
    result = analyzer.analyze_sentiment(test_text)
    print("\nTest sentiment analysis:")
    print(f"Text: {test_text}")
    print(f"Result: {result}")

    # Test batch
    test_texts = [
        "Company reports strong quarterly earnings beating expectations.",
        "Market crashes amid fears of recession and unemployment.",
        "The Federal Reserve maintains interest rates at current levels."
    ]
    results = analyzer.analyze_batch(test_texts)
    print("\nBatch sentiment analysis:")
    for text, result in zip(test_texts, results):
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']} (score: {result['score']:.3f})")
