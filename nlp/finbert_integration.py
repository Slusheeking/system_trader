#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FinBERT Integration for System Trader
------------------------------------
This module provides integration between the System Trader platform
and our rule-based financial sentiment analysis implementation.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Union

# Import our pure Python implementation
from . import run_finbert_pure
from .run_finbert_pure import analyze_text

class FinBERTAnalyzer:
    """
    A class that provides financial sentiment analysis using our rule-based approach.
    This class is designed to be easily integrated with the System Trader platform.
    """
    
    def __init__(self):
        """Initialize the FinBERT analyzer."""
        self.name = "FinBERT-RuleBased"
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary containing sentiment analysis results
        """
        return analyze_text(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of a batch of texts.
        
        Args:
            texts: A list of texts to analyze
            
        Returns:
            A list of dictionaries containing sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts]
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get just the normalized sentiment score for a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A float representing the sentiment score (-1 to 1)
        """
        result = self.analyze_sentiment(text)
        return result['normalized_score']
    
    def get_sentiment_label(self, text: str) -> str:
        """
        Get just the sentiment label for a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A string representing the sentiment ('positive', 'negative', or 'neutral')
        """
        result = self.analyze_sentiment(text)
        return result['sentiment']
    
    def analyze_news_item(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a news item dictionary.
        
        Args:
            news_item: A dictionary containing news information with at least a 'text' or 'headline' field
            
        Returns:
            The news item with added sentiment analysis fields
        """
        # Make a copy to avoid modifying the original
        result = news_item.copy()
        
        # Determine which field to analyze
        text = result.get('text', result.get('headline', result.get('content', '')))
        
        if text:
            sentiment_result = self.analyze_sentiment(text)
            result['sentiment'] = sentiment_result['sentiment']
            result['sentiment_score'] = sentiment_result['normalized_score']
            result['sentiment_details'] = {
                'positive_terms': sentiment_result['positive_terms'],
                'negative_terms': sentiment_result['negative_terms'],
                'positive_matches': sentiment_result.get('positive_matches', []),
                'negative_matches': sentiment_result.get('negative_matches', [])
            }
        else:
            result['sentiment'] = 'neutral'
            result['sentiment_score'] = 0.0
            result['sentiment_details'] = {
                'positive_terms': 0,
                'negative_terms': 0,
                'positive_matches': [],
                'negative_matches': []
            }
            
        return result

# Create a singleton instance for easy import
analyzer = FinBERTAnalyzer()

# Example usage
if __name__ == "__main__":
    # Test with a few examples
    test_texts = [
        "The company reported strong earnings, beating analyst expectations.",
        "The company missed earnings expectations and reported significant losses.",
        "The market remained stable with no significant changes."
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['normalized_score']:.4f}")
        print("-" * 50)