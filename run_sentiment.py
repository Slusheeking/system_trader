#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentiment Analysis Runner
------------------------
Command-line interface for running sentiment analysis without dependencies on the full System Trader.
"""

import argparse
import json
import sys
import os
import logging
from typing import Dict, List, Any, Optional

# Ensure log directory exists before setting up logging
def ensure_log_directory():
    """Ensure the logs directory exists."""
    os.makedirs('logs', exist_ok=True)

# Create log directory
ensure_log_directory()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/sentiment_analysis.log', mode='a')
    ]
)

logger = logging.getLogger('sentiment_analysis')

# Function moved to the top of the file

# Global variable to track if SentimentAnalyzer has been tried and failed
SENTIMENT_ANALYZER_FAILED = False
# Global analyzer instance
RULE_BASED_ANALYZER = None

def get_rule_based_analyzer():
    """Get or create the rule-based analyzer."""
    global RULE_BASED_ANALYZER
    if RULE_BASED_ANALYZER is None:
        from nlp.run_finbert_pure import analyze_text as pure_analyze_text
        RULE_BASED_ANALYZER = pure_analyze_text
    return RULE_BASED_ANALYZER

def analyze_text(text: str, force_rule_based: bool = False) -> Dict[str, Any]:
    """
    Analyze sentiment of a text.
    
    Args:
        text: Text to analyze
        force_rule_based: Whether to force using rule-based implementation
        
    Returns:
        Dictionary with sentiment analysis results
    """
    global SENTIMENT_ANALYZER_FAILED
    
    logger.info(f"Analyzing text: {text[:50]}...")
    
    try:
        # If we've already tried and failed with SentimentAnalyzer, or if force_rule_based is True,
        # use rule-based analyzer directly
        if force_rule_based or SENTIMENT_ANALYZER_FAILED:
            # Use rule-based analyzer
            if not SENTIMENT_ANALYZER_FAILED:
                logger.info("Using rule-based analyzer (forced)")
            
            pure_analyze_text = get_rule_based_analyzer()
            result = pure_analyze_text(text)
            
            # Convert to standard format
            return {
                'text': text,
                'classification': 'bullish' if result['sentiment'] == 'positive' else
                                ('bearish' if result['sentiment'] == 'negative' else 'neutral'),
                'sentiment': result['sentiment'],
                'score': result['normalized_score'],
                'magnitude': abs(result['normalized_score']),
                'details': {
                    'positive_terms': result['positive_terms'],
                    'negative_terms': result['negative_terms'],
                    'positive_matches': result.get('positive_matches', []),
                    'negative_matches': result.get('negative_matches', [])
                }
            }
        else:
            # Try to use SentimentAnalyzer
            logger.info("Attempting to use SentimentAnalyzer")
            try:
                from nlp.sentiment_analyzer import SentimentAnalyzer
                analyzer = SentimentAnalyzer(use_rule_based=True)
                return analyzer.analyze(text)
            except Exception as e:
                logger.error(f"Error using SentimentAnalyzer: {str(e)}")
                logger.info("Falling back to rule-based analyzer")
                
                # Mark SentimentAnalyzer as failed so we don't try again
                SENTIMENT_ANALYZER_FAILED = True
                
                # Fall back to rule-based
                pure_analyze_text = get_rule_based_analyzer()
                result = pure_analyze_text(text)
                
                # Convert to standard format
                return {
                    'text': text,
                    'classification': 'bullish' if result['sentiment'] == 'positive' else
                                    ('bearish' if result['sentiment'] == 'negative' else 'neutral'),
                    'sentiment': result['sentiment'],
                    'score': result['normalized_score'],
                    'magnitude': abs(result['normalized_score']),
                    'details': {
                        'positive_terms': result['positive_terms'],
                        'negative_terms': result['negative_terms'],
                        'positive_matches': result.get('positive_matches', []),
                        'negative_matches': result.get('negative_matches', [])
                    }
                }
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return {
            'text': text,
            'classification': 'error',
            'sentiment': 'error',
            'score': 0.0,
            'magnitude': 0.0,
            'error': str(e)
        }

def analyze_file(file_path: str, force_rule_based: bool = False) -> List[Dict[str, Any]]:
    """
    Analyze sentiment of texts in a file.
    
    Args:
        file_path: Path to file with texts (one per line)
        use_rule_based: Whether to use rule-based implementation
        
    Returns:
        List of dictionaries with sentiment analysis results
    """
    logger.info(f"Analyzing file: {file_path}")
    
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # Analyze each text
        results = []
        for text in texts:
            result = analyze_text(text, force_rule_based)
            results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        return []

def main():
    """Main entry point for the sentiment analysis runner."""
    # Log directory is already ensured at the start of the script
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis Runner')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File with texts to analyze (one per line)')
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--force-rule-based', action='store_true', help='Force using rule-based implementation')
    
    args = parser.parse_args()
    
    # Check if text or file is provided
    if not args.text and not args.file:
        parser.print_help()
        print("\nError: Either --text or --file must be provided")
        sys.exit(1)
    
    # Analyze text or file
    if args.text:
        result = analyze_text(args.text, args.force_rule_based)
        
        # Print result
        print("\nSentiment Analysis Result:")
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['classification']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Magnitude: {result['magnitude']:.4f}")
        
        if 'details' in result:
            details = result['details']
            print("\nDetails:")
            print(f"  Positive terms: {details['positive_terms']}")
            print(f"  Negative terms: {details['negative_terms']}")
            
            if details.get('positive_matches'):
                print(f"  Positive matches: {', '.join(details['positive_matches'])}")
            if details.get('negative_matches'):
                print(f"  Negative matches: {', '.join(details['negative_matches'])}")
        
        # Save result if output path provided
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to {args.output}")
    
    else:  # args.file
        results = analyze_file(args.file, args.force_rule_based)
        
        # Print summary
        print("\nSentiment Analysis Results:")
        print(f"Analyzed {len(results)} texts")
        
        # Count by sentiment
        sentiment_counts = {}
        for result in results:
            sentiment = result['classification']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} ({count/len(results):.1%})")
        
        # Print first 5 results
        print("\nSample Results:")
        for i, result in enumerate(results[:5]):
            print(f"  {i+1}. {result['text'][:50]}...")
            print(f"     Sentiment: {result['classification']}")
            print(f"     Score: {result['score']:.4f}")
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    logger.info("Sentiment analysis completed")

if __name__ == '__main__':
    main()