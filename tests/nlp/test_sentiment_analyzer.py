#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Sentiment Analysis Test
Tests only the sentiment analyzer with predefined text
"""

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sentiment_test")

try:
    from nlp.sentiment_analyzer import SentimentAnalyzer
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# Test text samples
TEST_TEXTS = [
    # Positive samples
    "I'm bullish on AAPL after seeing their latest earnings report. The growth is impressive.",
    "TSLA just announced new technology that will revolutionize the industry. Strong buy!",
    "MSFT's cloud business is growing rapidly and beating expectations. The future looks bright.",
    
    # Negative samples
    "NFLX is losing subscribers and facing increased competition. I'm bearish on their outlook.",
    "FB's advertising revenue is declining and they're facing regulatory challenges. Sell now.",
    "The airline industry (AAL, DAL) is facing headwinds with rising fuel costs and reduced travel.",
    
    # Neutral samples
    "AMZN reported quarterly results that met analyst expectations.",
    "The Fed's decision could impact interest rate sensitive stocks.",
    "Market volatility remains within expected ranges for SPY."
]

def test_sentiment_analysis():
    """Test sentiment analysis on predefined text samples."""
    try:
        # Initialize sentiment analyzers
        logger.info("Initializing rule-based sentiment analyzer...")
        rule_based = SentimentAnalyzer(use_finbert=False)
        
        # Test if FinBERT is available
        try:
            logger.info("Trying to initialize FinBERT sentiment analyzer...")
            finbert = SentimentAnalyzer(use_finbert=True)
            have_finbert = True
            logger.info("FinBERT initialization successful")
        except Exception as e:
            logger.warning(f"FinBERT not available: {str(e)}")
            have_finbert = False
        
        # Test sentiment analysis
        logger.info("\nTesting sentiment analysis on sample texts:")
        print("\n===== SENTIMENT ANALYSIS RESULTS =====")
        print(f"{'Text Sample':<60} | {'Rule-Based':<12} | {'FinBERT':<12}")
        print("-" * 90)
        
        for text in TEST_TEXTS:
            # Analyze with rule-based
            rb_result = rule_based.analyze(text)
            rb_score = rb_result['score']
            
            # Analyze with FinBERT if available
            if have_finbert:
                fb_result = finbert.analyze(text)
                fb_score = fb_result['score']
            else:
                fb_score = "N/A"
            
            # Display results
            sample = text[:57] + "..." if len(text) > 57 else text
            print(f"{sample:<60} | {rb_score:+.2f}      | {fb_score if isinstance(fb_score, str) else fb_score:+.2f} ")
        
        print("\n✅ Sentiment analysis test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_sentiment_analysis():
        sys.exit(0)
    else:
        print("\n❌ Sentiment analysis test failed!")
        sys.exit(1)