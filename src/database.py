from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()


class SentimentRecord(Base):
    """Database model for sentiment analysis records"""
    __tablename__ = 'sentiment_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    source = Column(String(50), nullable=False)
    headline = Column(Text, nullable=False)
    sentiment_label = Column(String(20), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    positive_score = Column(Float, nullable=False)
    negative_score = Column(Float, nullable=False)
    neutral_score = Column(Float, nullable=False)
    compound_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class DailyIndex(Base):
    """Database model for daily sentiment index"""
    __tablename__ = 'daily_indices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, unique=True, nullable=False)
    sentiment_index = Column(Float, nullable=False)
    avg_compound = Column(Float, nullable=False)
    avg_positive = Column(Float, nullable=False)
    avg_negative = Column(Float, nullable=False)
    avg_neutral = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    momentum = Column(Float, nullable=True)
    article_count = Column(Integer, nullable=False)
    label_distribution = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.now)


class SourceIndex(Base):
    """Database model for source-specific sentiment index"""
    __tablename__ = 'source_indices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    source = Column(String(50), nullable=False)
    sentiment_index = Column(Float, nullable=False)
    avg_compound = Column(Float, nullable=False)
    article_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class SentimentDatabase:
    """Database operations for sentiment data"""

    def __init__(self, db_path='sentiment_data.db'):
        """Initialize database connection"""
        self.engine = create_engine(f'sqlite:///{db_path}', pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._session = None

    @property
    def session(self):
        """Get or create a session with automatic rollback on error"""
        if self._session is None:
            self._session = self.Session()
        return self._session

    def _handle_error(self, error):
        """Handle database errors by rolling back the session"""
        if self._session:
            self._session.rollback()
        raise error

    def reset_session(self):
        """Close and reset the session"""
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def save_sentiment_records(self, df):
        """
        Save sentiment analysis results to database

        Args:
            df: DataFrame with sentiment analysis
        """
        try:
            records = []
            for _, row in df.iterrows():
                record = SentimentRecord(
                    date=row.get('date', datetime.now()),
                    source=row.get('source', 'Unknown'),
                    headline=row.get('text', '')[:1000],  # Truncate if too long
                    sentiment_label=row.get('sentiment_label', 'neutral'),
                    sentiment_score=row.get('sentiment_score', 0.0),
                    positive_score=row.get('positive_score', 0.0),
                    negative_score=row.get('negative_score', 0.0),
                    neutral_score=row.get('neutral_score', 0.0),
                    compound_score=row.get('compound_score', 0.0)
                )
                records.append(record)

            self.session.bulk_save_objects(records)
            self.session.commit()
            print(f"Saved {len(records)} sentiment records to database")
        except Exception as e:
            self._handle_error(e)

    def save_daily_index(self, daily_index_df):
        """
        Save daily sentiment indices to database

        Args:
            daily_index_df: DataFrame with daily indices
        """
        try:
            for _, row in daily_index_df.iterrows():
                # Check if record exists
                existing = self.session.query(DailyIndex).filter_by(date=row['date']).first()

                label_dist = row.get('label_distribution', {})
                if isinstance(label_dist, dict):
                    label_dist_json = json.dumps(label_dist)
                else:
                    label_dist_json = str(label_dist)

                if existing:
                    # Update existing record
                    existing.sentiment_index = row['sentiment_index']
                    existing.avg_compound = row['avg_compound']
                    existing.avg_positive = row['avg_positive']
                    existing.avg_negative = row['avg_negative']
                    existing.avg_neutral = row['avg_neutral']
                    existing.volatility = row['volatility']
                    existing.momentum = row.get('momentum', 0.0)
                    existing.article_count = row['article_count']
                    existing.label_distribution = label_dist_json
                else:
                    # Create new record
                    index_record = DailyIndex(
                        date=row['date'],
                        sentiment_index=row['sentiment_index'],
                        avg_compound=row['avg_compound'],
                        avg_positive=row['avg_positive'],
                        avg_negative=row['avg_negative'],
                        avg_neutral=row['avg_neutral'],
                        volatility=row['volatility'],
                        momentum=row.get('momentum', 0.0),
                        article_count=row['article_count'],
                        label_distribution=label_dist_json
                    )
                    self.session.add(index_record)

            self.session.commit()
            print(f"Saved {len(daily_index_df)} daily index records to database")
        except Exception as e:
            self._handle_error(e)

    def save_source_index(self, source_index_df, date=None):
        """
        Save source-specific sentiment indices

        Args:
            source_index_df: DataFrame with source indices
            date: Date for these indices
        """
        try:
            if date is None:
                date = datetime.now()

            for _, row in source_index_df.iterrows():
                index_record = SourceIndex(
                    date=date,
                    source=row['source'],
                    sentiment_index=row['sentiment_index'],
                    avg_compound=row['avg_compound'],
                    article_count=row['article_count']
                )
                self.session.add(index_record)

            self.session.commit()
            print(f"Saved {len(source_index_df)} source index records to database")
        except Exception as e:
            self._handle_error(e)

    def get_latest_index(self):
        """Get the most recent sentiment index"""
        try:
            latest = self.session.query(DailyIndex).order_by(DailyIndex.date.desc()).first()
            if latest:
                return {
                    'date': latest.date.isoformat(),
                    'sentiment_index': latest.sentiment_index,
                    'avg_compound': latest.avg_compound,
                    'volatility': latest.volatility,
                    'momentum': latest.momentum,
                    'article_count': latest.article_count,
                    'label_distribution': json.loads(latest.label_distribution) if latest.label_distribution else {}
                }
            return None
        except Exception as e:
            self._handle_error(e)

    def get_historical_index(self, days=30):
        """
        Get historical sentiment index data

        Args:
            days: Number of days to retrieve

        Returns:
            list: Historical index data
        """
        try:
            from datetime import timedelta
            start_date = datetime.now() - timedelta(days=days)

            records = self.session.query(DailyIndex).filter(
                DailyIndex.date >= start_date
            ).order_by(DailyIndex.date.asc()).all()

            return [{
                'date': r.date.isoformat(),
                'sentiment_index': r.sentiment_index,
                'avg_compound': r.avg_compound,
                'volatility': r.volatility,
                'momentum': r.momentum,
                'article_count': r.article_count
            } for r in records]
        except Exception as e:
            self._handle_error(e)

    def get_sentiment_by_source(self, days=7):
        """Get recent sentiment breakdown by source"""
        try:
            from datetime import timedelta
            start_date = datetime.now() - timedelta(days=days)

            records = self.session.query(SourceIndex).filter(
                SourceIndex.date >= start_date
            ).order_by(SourceIndex.date.desc()).all()

            return [{
                'date': r.date.isoformat(),
                'source': r.source,
                'sentiment_index': r.sentiment_index,
                'avg_compound': r.avg_compound,
                'article_count': r.article_count
            } for r in records]
        except Exception as e:
            self._handle_error(e)

    def get_latest_records(self, limit=50):
        """Get most recent sentiment records"""
        try:
            records = self.session.query(SentimentRecord).order_by(
                SentimentRecord.created_at.desc()
            ).limit(limit).all()

            return [{
                'date': r.date.isoformat(),
                'source': r.source,
                'headline': r.headline,
                'sentiment_label': r.sentiment_label,
                'sentiment_score': r.sentiment_score,
                'positive_score': r.positive_score,
                'negative_score': r.negative_score,
                'neutral_score': r.neutral_score,
                'compound_score': r.compound_score
            } for r in records]
        except Exception as e:
            self._handle_error(e)

    def close(self):
        """Close database connection"""
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            finally:
                self._session = None


if __name__ == "__main__":
    # Test database operations
    db = SentimentDatabase('test_sentiment.db')

    print("Database tables created successfully!")

    # Test data
    import pandas as pd

    test_df = pd.DataFrame([{
        'date': datetime.now(),
        'source': 'Test',
        'text': 'Test headline',
        'sentiment_label': 'positive',
        'sentiment_score': 0.95,
        'positive_score': 0.95,
        'negative_score': 0.02,
        'neutral_score': 0.03,
        'compound_score': 0.93
    }])

    db.save_sentiment_records(test_df)
    print("Test record saved!")

    db.close()
