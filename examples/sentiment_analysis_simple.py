#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Sentiment Analysis Example
--------------------------------
This example demonstrates how to use the sentiment analysis functionality
in the System Trader platform.
"""

import sys
import os
import json
from datetime import datetime, timedelta
import logging

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We'll only use the rule-based sentiment analysis to avoid NumPy compatibility issues
SENTIMENT_FEATURES_AVAILABLE = False
print("Using rule-based sentiment analysis only")

# Import rule-based sentiment analysis
try:
    from nlp.run_finbert_pure import analyze_text
    RULE_BASED_AVAILABLE = True
except ImportError:
    RULE_BASED_AVAILABLE = False
    print("Rule-based sentiment analysis not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sentiment_example')

def analyze_stock_sentiment(texts, stock_symbol):
    """
    Analyze sentiment for a stock based on a list of texts.
    
    Args:
        texts: List of texts to analyze
        stock_symbol: Stock symbol
        
    Returns:
        Dictionary with sentiment analysis results
    """
    logger.info(f"Analyzing sentiment for {stock_symbol} with {len(texts)} texts")
    
    # We're skipping ML-based sentiment analysis due to NumPy compatibility issues
    
    # Fall back to rule-based sentiment analysis
    if RULE_BASED_AVAILABLE:
        try:
            # Analyze each text
            results = []
            for text in texts:
                result = analyze_text(text)
                results.append(result)
            
            # Calculate average sentiment
            avg_score = sum(r['normalized_score'] for r in results) / len(results)
            avg_magnitude = sum(abs(r['normalized_score']) for r in results) / len(results)
            bullish_count = sum(1 for r in results if r['sentiment'] == 'positive')
            bearish_count = sum(1 for r in results if r['sentiment'] == 'negative')
            
            return {
                'symbol': stock_symbol,
                'sentiment_score': avg_score,
                'sentiment_magnitude': avg_magnitude,
                'bullish_ratio': bullish_count / len(results) if results else 0,
                'bearish_ratio': bearish_count / len(results) if results else 0,
                'method': 'rule_based'
            }
        except Exception as e:
            logger.error(f"Error using rule-based sentiment analysis: {str(e)}")
    
    # Return empty result if both methods fail
    return {
        'symbol': stock_symbol,
        'sentiment_score': 0,
        'sentiment_magnitude': 0,
        'bullish_ratio': 0,
        'bearish_ratio': 0,
        'method': 'none'
    }

def compare_stocks_sentiment(stock_texts):
    """
    Compare sentiment across multiple stocks.
    
    Args:
        stock_texts: Dictionary mapping stock symbols to lists of texts
        
    Returns:
        DataFrame with sentiment comparison
    """
    results = []
    
    for symbol, texts in stock_texts.items():
        sentiment = analyze_stock_sentiment(texts, symbol)
        results.append(sentiment)
    
    return results

def print_sentiment_comparison(results):
    """
    Print sentiment comparison across stocks.
    
    Args:
        results: List of dictionaries with sentiment results
    """
    # Print header
    print("\n{:<6} {:<15} {:<15} {:<15} {:<15} {:<10}".format(
        "Symbol", "Sentiment", "Magnitude", "Bullish %", "Bearish %", "Method"
    ))
    print("-" * 80)
    
    # Print each result
    for result in results:
        print("{:<6} {:<15.4f} {:<15.4f} {:<15.1%} {:<15.1%} {:<10}".format(
            result['symbol'],
            result['sentiment_score'],
            result['sentiment_magnitude'],
            result['bullish_ratio'],
            result['bearish_ratio'],
            result['method']
        ))
    
    # Print summary
    avg_sentiment = sum(r['sentiment_score'] for r in results) / len(results)
    print("\nAverage sentiment across all stocks: {:.4f}".format(avg_sentiment))
    
    # Print interpretation
    if avg_sentiment > 0.3:
        print("Overall market sentiment: BULLISH")
    elif avg_sentiment < -0.3:
        print("Overall market sentiment: BEARISH")
    else:
        print("Overall market sentiment: NEUTRAL")

def main():
    """Main function to demonstrate sentiment analysis."""
    # Example texts for different stocks
    stock_texts = {
        'AAPL': [
            "Apple's new iPhone 15 Pro has exceeded sales expectations, with analysts projecting record-breaking revenue.",
            "The company's services division continues to show strong growth, with Apple TV+ gaining market share.",
            "Supply chain issues have been resolved, leading to improved production capacity for the holiday season."
        ],
        'MSFT': [
            "Microsoft's cloud business Azure reported 27% growth, slightly below analyst expectations.",
            "The company announced layoffs in its gaming division following the Activision Blizzard acquisition.",
            "Microsoft's AI investments are starting to pay off with new features in Office and Windows."
        ],
        'GOOGL': [
            "Google's ad revenue declined for the second consecutive quarter amid increased competition.",
            "The company's AI search features have received mixed reviews from users and publishers.",
            "Regulatory concerns continue to weigh on Alphabet's stock as antitrust scrutiny intensifies."
        ],
        'AMZN': [
            "Amazon Web Services showed strong growth, offsetting weakness in the retail segment.",
            "The company's cost-cutting measures have improved margins and operating efficiency.",
            "Amazon's Prime membership continues to grow, with increased engagement across shopping and streaming."
        ]
    }
    
    # Compare sentiment across stocks
    results = compare_stocks_sentiment(stock_texts)
    
    # Print results
    print("\nSentiment Analysis Results:")
    print_sentiment_comparison(results)
    
    # Save results to file
    try:
        with open('sentiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to sentiment_results.json")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    main()