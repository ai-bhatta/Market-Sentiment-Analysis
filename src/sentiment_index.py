import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SentimentIndexCalculator:
    """Calculate aggregate sentiment indices from news data"""

    def __init__(self):
        self.base_index = 100  # Base index value

    def calculate_daily_index(self, df, date_column='date'):
        """
        Calculate daily sentiment index

        Args:
            df: DataFrame with sentiment scores
            date_column: Name of date column

        Returns:
            DataFrame: Daily aggregated sentiment index
        """
        if date_column not in df.columns or df[date_column].isna().all():
            print("Warning: No valid dates found, using overall sentiment")
            return self._calculate_overall_index(df)

        # Filter out rows with no date
        df_dated = df[df[date_column].notna()].copy()

        if len(df_dated) == 0:
            return self._calculate_overall_index(df)

        # Group by date
        daily_stats = df_dated.groupby(df_dated[date_column].dt.date).agg({
            'compound_score': ['mean', 'std', 'count'],
            'positive_score': 'mean',
            'negative_score': 'mean',
            'neutral_score': 'mean',
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).reset_index()

        # Flatten column names
        daily_stats.columns = ['date', 'avg_compound', 'std_compound', 'article_count',
                               'avg_positive', 'avg_negative', 'avg_neutral', 'label_distribution']

        # Calculate sentiment index (scale from 0-200, 100 is neutral)
        daily_stats['sentiment_index'] = self.base_index + (daily_stats['avg_compound'] * 50)

        # Calculate volatility (standard deviation of sentiment)
        daily_stats['volatility'] = daily_stats['std_compound'].fillna(0)

        # Calculate momentum (change from previous day)
        daily_stats['momentum'] = daily_stats['sentiment_index'].diff()

        return daily_stats.sort_values('date')

    def calculate_source_index(self, df, source_column='source'):
        """
        Calculate sentiment index by news source

        Args:
            df: DataFrame with sentiment scores
            source_column: Name of source column

        Returns:
            DataFrame: Sentiment index by source
        """
        source_stats = df.groupby(source_column).agg({
            'compound_score': ['mean', 'std', 'count'],
            'positive_score': 'mean',
            'negative_score': 'mean',
            'neutral_score': 'mean',
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).reset_index()

        # Flatten column names
        source_stats.columns = ['source', 'avg_compound', 'std_compound', 'article_count',
                                'avg_positive', 'avg_negative', 'avg_neutral', 'label_distribution']

        # Calculate sentiment index
        source_stats['sentiment_index'] = self.base_index + (source_stats['avg_compound'] * 50)

        return source_stats

    def calculate_overall_index(self, df):
        """
        Calculate overall market sentiment index

        Args:
            df: DataFrame with sentiment scores

        Returns:
            dict: Overall sentiment metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_articles': len(df),
            'avg_compound_score': df['compound_score'].mean(),
            'avg_positive_score': df['positive_score'].mean(),
            'avg_negative_score': df['negative_score'].mean(),
            'avg_neutral_score': df['neutral_score'].mean(),
            'sentiment_index': self.base_index + (df['compound_score'].mean() * 50),
            'volatility': df['compound_score'].std(),
            'sentiment_distribution': df['sentiment_label'].value_counts().to_dict()
        }

        # Calculate bullish/bearish ratio
        positive_count = (df['sentiment_label'] == 'positive').sum()
        negative_count = (df['sentiment_label'] == 'negative').sum()

        if negative_count > 0:
            metrics['bullish_bearish_ratio'] = positive_count / negative_count
        else:
            metrics['bullish_bearish_ratio'] = float('inf') if positive_count > 0 else 1.0

        return metrics

    def _calculate_overall_index(self, df):
        """Helper to calculate overall when no dates available"""
        overall = self.calculate_overall_index(df)
        return pd.DataFrame([{
            'date': datetime.now().date(),
            'avg_compound': overall['avg_compound_score'],
            'std_compound': overall['volatility'],
            'article_count': overall['total_articles'],
            'avg_positive': overall['avg_positive_score'],
            'avg_negative': overall['avg_negative_score'],
            'avg_neutral': overall['avg_neutral_score'],
            'sentiment_index': overall['sentiment_index'],
            'volatility': overall['volatility'],
            'momentum': 0.0,
            'label_distribution': overall['sentiment_distribution']
        }])

    def calculate_moving_average(self, daily_index_df, window=7):
        """
        Calculate moving average of sentiment index

        Args:
            daily_index_df: DataFrame with daily indices
            window: Moving average window size

        Returns:
            DataFrame: With moving average column added
        """
        df = daily_index_df.copy()
        df[f'ma_{window}'] = df['sentiment_index'].rolling(window=window, min_periods=1).mean()
        return df

    def interpret_index(self, index_value):
        """
        Interpret sentiment index value

        Args:
            index_value: Sentiment index value

        Returns:
            str: Interpretation of the index
        """
        if index_value >= 120:
            return "Very Bullish"
        elif index_value >= 110:
            return "Bullish"
        elif index_value >= 90:
            return "Neutral"
        elif index_value >= 80:
            return "Bearish"
        else:
            return "Very Bearish"

    def generate_summary_report(self, df):
        """
        Generate comprehensive sentiment summary

        Args:
            df: DataFrame with sentiment analysis

        Returns:
            dict: Summary report
        """
        overall = self.calculate_overall_index(df)
        daily = self.calculate_daily_index(df)
        by_source = self.calculate_source_index(df)

        report = {
            'overall_metrics': overall,
            'interpretation': self.interpret_index(overall['sentiment_index']),
            'daily_trend': daily.to_dict('records'),
            'source_breakdown': by_source.to_dict('records'),
            'latest_index': overall['sentiment_index'],
            'latest_date': datetime.now().isoformat()
        }

        return report


if __name__ == "__main__":
    # Test the index calculator
    from data_loader import NewsDataLoader
    from sentiment_analyzer import FinBERTSentimentAnalyzer

    print("Loading data...")
    loader = NewsDataLoader()
    df = loader.load_all_data()

    print("\nAnalyzing sentiment...")
    analyzer = FinBERTSentimentAnalyzer()
    df = analyzer.analyze_dataframe(df.head(100))  # Test with first 100

    print("\nCalculating indices...")
    calculator = SentimentIndexCalculator()

    overall = calculator.calculate_overall_index(df)
    print("\nOverall Sentiment Index:")
    print(f"Index Value: {overall['sentiment_index']:.2f}")
    print(f"Interpretation: {calculator.interpret_index(overall['sentiment_index'])}")

    daily = calculator.calculate_daily_index(df)
    print("\nDaily Sentiment Index:")
    print(daily.head())

    by_source = calculator.calculate_source_index(df)
    print("\nSentiment by Source:")
    print(by_source)
