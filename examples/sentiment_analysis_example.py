#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analysis Example
-------------------------
This example demonstrates how to use the sentiment analysis capabilities
of the System Trader platform, including the rule-based FinBERT implementation.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sentiment analyzer
from nlp.sentiment_analyzer import SentimentAnalyzer

def analyze_text_samples():
    """
    Analyze a set of financial text samples using different sentiment analyzers.
    """
    print("\n=== Analyzing Financial Text Samples ===\n")
    
    # Sample financial texts
    samples = [
        "The company reported strong earnings, beating analyst expectations.",
        "The company missed earnings expectations and reported significant losses.",
        "The market remained stable with no significant changes.",
        "Investors are optimistic about the company's growth prospects.",
        "The stock plummeted after the CEO resigned amid accounting irregularities.",
        "The merger is expected to create significant synergies and cost savings.",
        "Regulatory concerns have cast a shadow over the company's expansion plans.",
        "The company announced a major share buyback program."
    ]
    
    # Initialize analyzers
    vader_only = SentimentAnalyzer(use_finbert=False)
    rule_based = SentimentAnalyzer(use_rule_based=True)
    
    # Try ML-based FinBERT (may fail due to NumPy compatibility)
    try:
        ml_based = SentimentAnalyzer(use_finbert=True)
        ml_based_available = True
    except Exception as e:
        print(f"ML-based FinBERT not available: {str(e)}")
        ml_based_available = False
    
    # Analyze samples with each analyzer
    results = []
    for sample in samples:
        result = {
            'text': sample,
            'vader': vader_only.analyze(sample),
            'rule_based': rule_based.analyze(sample)
        }
        
        if ml_based_available:
            result['ml_based'] = ml_based.analyze(sample)
        
        results.append(result)
        
        # Print results
        print(f"Text: {sample}")
        print(f"VADER: {result['vader']['classification']} ({result['vader']['score']:.2f})")
        print(f"Rule-based: {result['rule_based']['classification']} ({result['rule_based']['score']:.2f})")
        
        if ml_based_available:
            print(f"ML-based: {result['ml_based']['classification']} ({result['ml_based']['score']:.2f})")
        
        print("-" * 80)
    
    return results

def compare_sentiment_for_stocks():
    """
    Compare sentiment for different stocks using mock data.
    """
    print("\n=== Comparing Sentiment for Different Stocks ===\n")
    
    # Mock news data for different stocks
    stock_news = {
        'AAPL': [
            "Apple reports record iPhone sales, beating analyst expectations.",
            "Apple's new product launch receives positive reviews from critics.",
            "Apple's services revenue continues to grow at an impressive rate."
        ],
        'MSFT': [
            "Microsoft cloud business shows strong growth in latest quarter.",
            "Microsoft announces new AI features for its productivity suite.",
            "Microsoft's gaming division sees slight decline in revenue."
        ],
        'GOOGL': [
            "Google faces antitrust scrutiny over its advertising business.",
            "Google's parent company Alphabet reports better than expected earnings.",
            "Google's new AI model outperforms competitors in benchmark tests."
        ],
        'AMZN': [
            "Amazon misses revenue expectations as online sales slow.",
            "Amazon Web Services continues to be the profit engine for the company.",
            "Amazon announces layoffs amid economic uncertainty."
        ]
    }
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(use_rule_based=True)
    
    # Analyze sentiment for each stock
    stock_sentiment = {}
    for ticker, news_items in stock_news.items():
        sentiment_scores = [analyzer.analyze(news)['score'] for news in news_items]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        stock_sentiment[ticker] = {
            'news': news_items,
            'scores': sentiment_scores,
            'average': avg_sentiment
        }
        
        print(f"Stock: {ticker}")
        print(f"Average Sentiment: {avg_sentiment:.2f}")
        for i, (news, score) in enumerate(zip(news_items, sentiment_scores)):
            print(f"  News {i+1}: {news[:50]}... Score: {score:.2f}")
        print("-" * 80)
    
    # Visualize results
    tickers = list(stock_sentiment.keys())
    avg_scores = [stock_sentiment[ticker]['average'] for ticker in tickers]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tickers, avg_scores, color=['g' if s > 0 else 'r' for s in avg_scores])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Average Sentiment Score by Stock')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.ylim(-1, 1)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('stock_sentiment_comparison.png')
    print("Visualization saved as 'stock_sentiment_comparison.png'")
    
    return stock_sentiment

def analyze_sentiment_over_time():
    """
    Analyze sentiment over time for a stock using mock data.
    """
    print("\n=== Analyzing Sentiment Over Time ===\n")
    
    # Mock news data for a stock over time
    dates = pd.date_range(start='2025-01-01', periods=10, freq='W')
    news_data = [
        "Company announces new product line, stock rises 2%.",
        "Quarterly earnings beat expectations, outlook positive.",
        "CEO interviewed on financial news, highlights growth strategy.",
        "Company faces supply chain challenges, may impact next quarter.",
        "New partnership announced with major tech company.",
        "Analyst downgrades stock citing valuation concerns.",
        "Company completes acquisition of smaller competitor.",
        "Regulatory investigation announced, stock drops 5%.",
        "Company announces share buyback program.",
        "New product launch receives mixed reviews from industry experts."
    ]
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(use_rule_based=True)
    
    # Analyze sentiment for each news item
    sentiment_data = []
    for date, news in zip(dates, news_data):
        sentiment = analyzer.analyze(news)
        sentiment_data.append({
            'date': date,
            'news': news,
            'score': sentiment['score'],
            'classification': sentiment['classification']
        })
        
        print(f"Date: {date.strftime('%Y-%m-%d')}")
        print(f"News: {news}")
        print(f"Sentiment: {sentiment['classification']} ({sentiment['score']:.2f})")
        print("-" * 80)
    
    # Convert to DataFrame for easier visualization
    df = pd.DataFrame(sentiment_data)
    
    # Visualize sentiment over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['score'], marker='o', linestyle='-', color='b')
    plt.fill_between(df['date'], df['score'], 0, where=(df['score'] >= 0), color='g', alpha=0.3)
    plt.fill_between(df['date'], df['score'], 0, where=(df['score'] < 0), color='r', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Sentiment Score Over Time')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.ylim(-1, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_over_time.png')
    print("Visualization saved as 'sentiment_over_time.png'")
    
    return df

if __name__ == "__main__":
    print("=== System Trader Sentiment Analysis Example ===")
    
    # Run examples
    text_results = analyze_text_samples()
    stock_results = compare_sentiment_for_stocks()
    time_results = analyze_sentiment_over_time()
    
    print("\nAll examples completed successfully!")
