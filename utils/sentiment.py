import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

def get_market_sentiment(ticker):
    """
    Fetches latest news for a ticker and calculates a compound sentiment score.
    Returns:
        - sentiment_score: float (-1 to 1)
        - top_news: list of dicts {title, link, source}
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0.0, []

        analyzer = SentimentIntensityAnalyzer()
        scores = []
        top_news = []

        for item in news[:7]: # Take top 7 news items
            # Handle new yfinance news structure (nested under 'content')
            content = item.get('content', item) if 'content' in item else item
            
            title = content.get('title', '')
            summary = content.get('summary', '') or content.get('description', '')
            
            # Use both title and summary for sentiment if available
            text_to_analyze = f"{title}. {summary}" if summary else title
            
            if not title:
                continue

            score = analyzer.polarity_scores(text_to_analyze)['compound']
            scores.append(score)
            
            # Extract link and publisher with fallbacks for different versions
            link = content.get('clickThroughUrl', {}).get('url', content.get('link', '#'))
            publisher = content.get('provider', {}).get('displayName', content.get('publisher', 'Unknown'))
            
            top_news.append({
                'title': title,
                'link': link,
                'publisher': publisher
            })

        avg_sentiment = np.mean(scores) if scores else 0.0
        return avg_sentiment, top_news
    except Exception as e:
        print(f"Error fetching sentiment for {ticker}: {e}")
        return 0.0, []
