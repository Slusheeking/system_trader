# Sentiment Analysis in System Trader

This document explains the sentiment analysis capabilities in the System Trader platform, including architecture, data collection, feature generation, and integration with trading strategies.

## Architecture Overview

The sentiment analysis system in System Trader is designed with a modular, multi-layered approach:

1. **Data Collection Layer**: Gathers text data from various sources (financial news, social media, options flow)
2. **Processing Layer**: Cleans, normalizes, and prepares text data for analysis
3. **Analysis Layer**: Applies sentiment analysis techniques to extract sentiment scores
4. **Feature Generation Layer**: Converts raw sentiment data into tradable features
5. **Integration Layer**: Combines sentiment features with other signals in trading strategies

The system supports multiple sentiment analysis methods to provide robust results:

- **Rule-based Analysis**: Using lexicon-based approaches like VADER with financial enhancements
- **Machine Learning Analysis**: Using FinBERT, a financial domain-specific BERT model
- **Pure Python Implementation**: A fallback method for environments with compatibility issues
- **Command-line Interface**: A standalone tool for quick sentiment analysis without dependencies

## Data Collection

The sentiment analysis system collects data from multiple sources:

### Financial News
- Major financial news outlets (via Polygon API)
- Company press releases
- Analyst reports
- SEC filings

### Social Media
- Reddit (via PRAW)
- Twitter/X (via API)
- StockTwits

### Options Flow
- Unusual options activity
- Options volume and open interest
- Put/call ratios

## Sentiment Analysis Methods

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Rule-based sentiment analysis tool specifically attuned to social media
- Enhanced with financial domain-specific terms
- Provides compound scores from -1 (very negative) to +1 (very positive)

### FinBERT
- Financial domain-specific BERT model
- Pre-trained on financial communications
- Provides more accurate sentiment analysis for financial texts
- Available in two implementations:
  - ML-based: Using the transformers library (may have compatibility issues)
  - Rule-based: Pure Python implementation for environments with compatibility issues

### Composite Scoring
- Weighted combination of multiple sentiment analysis methods
- Adjustable weights based on source reliability and relevance
- Time-weighted to prioritize recent information

## Feature Generation

The sentiment analysis system generates several types of features:

### Basic Sentiment Features
- Raw sentiment scores (-1 to +1)
- Sentiment classifications (bullish, bearish, neutral)
- Sentiment magnitude (strength of sentiment)

### Advanced Sentiment Features
- Sentiment momentum (change in sentiment over time)
- Sentiment volatility (variability in sentiment)
- Sentiment divergence (difference between sources)
- Sentiment surprise (deviation from expected sentiment)

### Source-Specific Features

#### News Features
- Headline sentiment vs. content sentiment
- Publication credibility weighting
- News relevance scoring

#### Social Media Features
- Platform-specific sentiment (Reddit, Twitter, StockTwits)
- User reputation weighting
- Engagement metrics (upvotes, comments, shares)
- Sentiment dispersion across platforms

#### Options Flow Features
- Options sentiment indicators
- Put/call sentiment ratio
- Unusual activity sentiment

## Integration with Trading Strategies

Sentiment features can be integrated into trading strategies in several ways:

### Direct Signal Generation
- Generate buy/sell signals based on sentiment thresholds
- Use sentiment reversals as entry/exit triggers
- Filter trades based on sentiment confirmation

### Feature Enhancement
- Combine sentiment with technical indicators
- Use sentiment to adjust position sizing
- Apply sentiment filters to existing strategies

### Risk Management
- Adjust stop-loss levels based on sentiment
- Modify profit targets based on sentiment strength
- Use sentiment to determine market regime

## Implementation Details

### Sentiment Analyzer Class

The `SentimentAnalyzer` class in `nlp/sentiment_analyzer.py` provides the main interface for sentiment analysis:

```python
# Initialize with different options
analyzer = SentimentAnalyzer(use_finbert=True)  # ML-based FinBERT
analyzer = SentimentAnalyzer(use_rule_based=True)  # Rule-based implementation
analyzer = SentimentAnalyzer()  # VADER only

# Analyze a single text
result = analyzer.analyze("Apple reports strong earnings, beating expectations.")
print(result['score'])  # Sentiment score (-1 to +1)
print(result['classification'])  # 'bullish', 'bearish', or 'neutral'

# Analyze multiple texts
results = analyzer.batch_analyze(["Text 1", "Text 2", "Text 3"])
```

### Rule-Based FinBERT Implementation

The rule-based implementation in `nlp/finbert_integration.py` provides a pure Python alternative to the ML-based FinBERT:

```python
from nlp.finbert_integration import analyzer

# Analyze text
result = analyzer.analyze_sentiment("Apple reports strong earnings")
print(result['sentiment'])  # 'positive', 'negative', or 'neutral'
print(result['normalized_score'])  # Score from -1 to +1

# Get just the score
score = analyzer.get_sentiment_score("Apple reports strong earnings")

# Analyze a news item dictionary
news_item = {
    'headline': 'Apple Reports Q1 Results',
    'text': 'Apple reported strong earnings, beating analyst expectations.'
}
result = analyzer.analyze_news_item(news_item)
```

## Configuration

Sentiment analysis can be configured in `config/collector_config.yaml`:

```yaml
sentiment_analysis:
  enabled: true
  methods:
    vader:
      enabled: true
      weight: 0.4
    finbert:
      enabled: true
      weight: 0.6
      use_rule_based: true  # Fall back to rule-based if ML-based fails
  sources:
    news:
      enabled: true
      weight: 0.5
    social:
      enabled: true
      weight: 0.3
      platforms:
        reddit:
          enabled: true
          subreddits: ["wallstreetbets", "stocks", "investing"]
    options:
      enabled: true
      weight: 0.2
```

## Examples

### Full Example

See `examples/sentiment_analysis_example.py` for complete examples of:

1. Analyzing individual text samples
2. Comparing sentiment across multiple stocks
3. Tracking sentiment over time
4. Visualizing sentiment data

### Simple Example

For a simpler example that doesn't require pandas or matplotlib, see `examples/sentiment_analysis_simple.py`:

```python
# Import rule-based sentiment analysis
from nlp.run_finbert_pure import analyze_text

# Analyze a single text
result = analyze_text("Apple reports strong earnings, beating expectations.")
print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['normalized_score']}")

# Analyze multiple texts for a stock
texts = [
    "Company reported strong earnings",
    "New product launch was successful",
    "CEO announced expansion plans"
]

results = [analyze_text(text) for text in texts]
avg_score = sum(r['normalized_score'] for r in results) / len(results)
print(f"Average sentiment: {avg_score}")
```

### Command-line Interface

For quick sentiment analysis without dependencies, use the `run_sentiment.py` script:

```bash
# Analyze a single text
./run_sentiment.py --text "Apple reports strong earnings, beating expectations."

# Analyze multiple texts from a file (one per line)
./run_sentiment.py --file news_texts.txt

# Save results to a JSON file
./run_sentiment.py --file news_texts.txt --output results.json

# Force using rule-based implementation
./run_sentiment.py --text "Some text" --force-rule-based
```

## Troubleshooting

### NumPy Compatibility Issues

If you encounter NumPy compatibility errors like:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

Use the rule-based implementation instead:

```python
analyzer = SentimentAnalyzer(use_rule_based=True)
```

### Missing Dependencies

If you encounter missing dependencies, install the required packages:

```bash
pip install nltk textblob transformers torch praw
```

For the rule-based implementation only:

```bash
pip install nltk textblob
```

## Integration with ML Training Engine

The sentiment analysis functionality is integrated with the ML training engine, allowing ML models to use sentiment features:

```python
# In ml_training_engine_modified.py
def generate_sentiment_features(texts):
    """Generate sentiment features for a list of texts."""
    analyzer = get_sentiment_analyzer()
    features = []
    for text in texts:
        result = analyzer.analyze(text)
        features.append({
            'sentiment_score': result['score'],
            'sentiment_magnitude': result['magnitude'],
            'sentiment_is_bullish': 1 if result['classification'] == 'bullish' else 0,
            'sentiment_is_bearish': 1 if result['classification'] == 'bearish' else 0
        })
    return features
```

## Integration with Trading Strategies

A new sentiment-based trading strategy has been added to `trading/strategy.py`:

```python
class SentimentBasedStrategy(TradingStrategy):
    """Sentiment-based trading strategy."""
    
    def process_signals(self, signals):
        # Get sentiment data
        news_sentiment = position.get('news_sentiment', 0)
        social_sentiment = position.get('social_sentiment', 0)
        
        # Calculate composite sentiment
        composite_sentiment = (
            (news_sentiment * self.news_weight) +
            (social_sentiment * self.social_weight)
        )
        
        # Make trading decisions based on sentiment
        if composite_sentiment >= self.sentiment_threshold:
            # Generate buy signal
```

## Future Enhancements

Planned enhancements to the sentiment analysis system:

1. Support for more data sources (Bloomberg, Thomson Reuters, etc.)
2. Enhanced entity recognition to better identify company-specific sentiment
3. Aspect-based sentiment analysis to identify sentiment toward specific aspects (revenue, products, management)
4. Improved temporal analysis to better capture sentiment trends
5. Integration with alternative data sources (satellite imagery, credit card data)
6. Better handling of NumPy compatibility issues
7. Enhanced visualization tools for sentiment analysis
