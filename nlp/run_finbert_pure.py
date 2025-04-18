#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Financial Sentiment Analysis - Pure Python Implementation
-------------------------------------------------------
This script provides a simple rule-based financial sentiment analysis
without relying on any ML libraries to avoid NumPy compatibility issues.
"""

import argparse
import json
import sys
import os
import re
from typing import Dict, List, Any, Optional, Union

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of a text using a rule-based approach.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Define financial sentiment lexicons
    positive_terms = [
        'beat', 'beats', 'exceeded', 'exceed', 'better', 'strong', 'strongly', 'strength',
        'profit', 'profitable', 'profitability', 'growth', 'growing', 'grew', 'increase',
        'increased', 'increasing', 'gain', 'gains', 'gained', 'positive', 'advantage',
        'advantages', 'opportunity', 'opportunities', 'outperform', 'outperformed',
        'outperforming', 'outperforms', 'rise', 'rises', 'rising', 'rose', 'up', 'upside',
        'uptrend', 'higher', 'highest', 'high', 'record', 'boost', 'boosted', 'boosting',
        'boosts', 'improve', 'improved', 'improvement', 'improving', 'improves',
        'recovery', 'recovered', 'recovering', 'recovers', 'bullish', 'buy', 'buying'
    ]
    
    negative_terms = [
        'miss', 'missed', 'misses', 'missing', 'weak', 'weakness', 'weakening',
        'weakened', 'weaker', 'loss', 'losses', 'losing', 'lost', 'decline', 'declined',
        'declining', 'declines', 'decrease', 'decreased', 'decreasing', 'decreases',
        'negative', 'disadvantage', 'disadvantages', 'risk', 'risks', 'risky',
        'underperform', 'underperformed', 'underperforming', 'underperforms',
        'fall', 'falls', 'falling', 'fell', 'down', 'downside', 'downtrend',
        'lower', 'lowest', 'low', 'drop', 'drops', 'dropping', 'dropped',
        'worsen', 'worsened', 'worsening', 'worsens', 'worse', 'worst',
        'concern', 'concerns', 'concerning', 'concerned', 'bearish', 'sell', 'selling'
    ]
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count occurrences of positive and negative terms
    positive_count = sum(1 for term in positive_terms if term in text_lower)
    negative_count = sum(1 for term in negative_terms if term in text_lower)
    
    # Find all matching terms for explanation
    positive_matches = [term for term in positive_terms if term in text_lower]
    negative_matches = [term for term in negative_terms if term in text_lower]
    
    # Calculate sentiment score
    total_count = positive_count + negative_count
    if total_count == 0:
        sentiment = "neutral"
        score = 0.0
    else:
        positive_ratio = positive_count / total_count
        negative_ratio = negative_count / total_count
        
        if positive_ratio > negative_ratio:
            sentiment = "positive"
            score = positive_ratio
        elif negative_ratio > positive_ratio:
            sentiment = "negative"
            score = -negative_ratio
        else:
            sentiment = "neutral"
            score = 0.0
    
    # Normalize score to -1 to 1 range
    normalized_score = score
    
    return {
        'text': text,
        'sentiment': sentiment,
        'positive_terms': positive_count,
        'negative_terms': negative_count,
        'positive_matches': positive_matches,
        'negative_matches': negative_matches,
        'score': score,
        'normalized_score': normalized_score
    }

def analyze_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Analyze sentiment of texts in a file.
    
    Args:
        file_path: Path to file with texts (one per line)
        
    Returns:
        List of dictionaries with sentiment analysis results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = []
        for text in texts:
            result = analyze_text(text)
            results.append(result)
        
        return results
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Run financial sentiment analysis')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File with texts to analyze (one per line)')
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.print_help()
        print("\nError: Either --text or --file must be provided")
        sys.exit(1)
    
    # Analyze text or file
    if args.text:
        results = [analyze_text(args.text)]
    else:
        results = analyze_file(args.file)
    
    # Print results
    for result in results:
        print(f"Text: {result['text'][:50]}...")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Positive terms: {result['positive_terms']}")
        print(f"Negative terms: {result['negative_terms']}")
        if result['positive_matches']:
            print(f"Positive matches: {', '.join(result['positive_matches'])}")
        if result['negative_matches']:
            print(f"Negative matches: {', '.join(result['negative_matches'])}")
        print(f"Score: {result['score']:.4f}")
        print(f"Normalized Score (-1 to 1): {result['normalized_score']:.4f}")
        print("-" * 50)
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    print("Running pure Python financial sentiment analysis")
    main()