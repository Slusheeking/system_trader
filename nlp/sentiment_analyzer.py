#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analyzer Module
-------------------------
This module provides text sentiment analysis for trading signals
using both rule-based and ML-based approaches.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import numpy as np

# Import NLP libraries
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    # Download VADER lexicon if not already present
    nltk.download('vader_lexicon', quiet=True)
except ImportError:
    print("NLTK not installed. pip install nltk for VADER sentiment analysis.")

# Set environment variables to disable PyTorch features that might cause issues
import os
os.environ["ENABLE_FINBERT"] = "0"  # Disable ML-based FinBERT by default
os.environ["PYTORCH_JIT"] = "0"     # Disable PyTorch JIT

# Don't try to import torch by default
TRANSFORMERS_AVAILABLE = False

# Try to import transformers for ML-based sentiment analysis
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not installed. pip install transformers for ML-based sentiment analysis.")

# Try to import our rule-based FinBERT implementation
try:
    from nlp.finbert_integration import analyzer as rule_based_finbert
    RULE_BASED_FINBERT_AVAILABLE = True
except ImportError:
    RULE_BASED_FINBERT_AVAILABLE = False

from utils.logging import setup_logger

# Setup logging
logger = setup_logger('sentiment_analyzer', category='nlp')


class SentimentAnalyzer:
    """
    Sentiment analyzer for financial text using multiple models.
    
    This class combines rule-based (VADER) and ML-based (FinBERT) approaches
    to get more accurate sentiment scores for financial news and social media.
    """
    
    def __init__(self, use_finbert: bool = False, model_path: Optional[str] = None,
                 use_rule_based: bool = True):  # Default to rule-based for reliability
        """
        Initialize the sentiment analyzer.
        
        Args:
            use_finbert: Whether to use FinBERT model (requires more resources)
            model_path: Path to pre-trained FinBERT model (if None, will download)
            use_rule_based: Whether to use rule-based FinBERT implementation instead of ML-based
        """
        self.logger = logger
        
        # Initialize VADER (rule-based sentiment analyzer)
        try:
            self.vader = SentimentIntensityAnalyzer()
            self.logger.info("Initialized VADER sentiment analyzer")
        except Exception as e:
            self.logger.error(f"Failed to initialize VADER: {str(e)}")
            self.vader = None
        
        # Initialize FinBERT if requested
        self.use_finbert = use_finbert
        self.use_rule_based = use_rule_based
        self.finbert = None
        self.tokenizer = None
        
        # Check if rule-based implementation is requested and available
        if (use_rule_based or not TRANSFORMERS_AVAILABLE) and RULE_BASED_FINBERT_AVAILABLE:
            self.logger.info("Using rule-based FinBERT implementation")
            self.use_rule_based = True
            self.use_finbert = False  # Don't use ML-based if rule-based is requested
        elif use_finbert and TRANSFORMERS_AVAILABLE:
            try:
                # Use a more cautious approach to load models
                if model_path:
                    # Load from local path
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.finbert = AutoModelForSequenceClassification.from_pretrained(model_path)
                else:
                    # Load from Hugging Face
                    self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                    self.finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                
                # Set to evaluation mode
                if self.finbert:
                    self.finbert.eval()
                
                self.logger.info("Initialized FinBERT sentiment analyzer")
            except Exception as e:
                self.logger.error(f"Failed to initialize FinBERT: {str(e)}")
                self.use_finbert = False
                
                # Fall back to rule-based if available
                if RULE_BASED_FINBERT_AVAILABLE:
                    self.logger.info("Falling back to rule-based FinBERT implementation")
                    self.use_rule_based = True
        
        # Financial terms dictionaries for rule-based enhancement
        self.bullish_terms = {
            'bullish', 'outperform', 'buy', 'strong buy', 'upgrade', 'upside',
            'growth', 'positive', 'beat', 'exceeded', 'surge', 'rally', 'gain',
            'profit', 'optimistic', 'momentum', 'uptrend', 'breakout', 'opportunity'
        }
        
        self.bearish_terms = {
            'bearish', 'underperform', 'sell', 'strong sell', 'downgrade', 'downside',
            'decline', 'negative', 'miss', 'disappointed', 'plunge', 'drop', 'loss',
            'pessimistic', 'downtrend', 'breakdown', 'risk', 'warning'
        }
        
        # Financial and trading-specific terms to pay special attention to
        self.financial_terms = {
            'earnings', 'revenue', 'guidance', 'forecast', 'outlook', 'margin',
            'eps', 'profit', 'dividend', 'acquisition', 'merger', 'takeover',
            'bankruptcy', 'restructuring', 'lawsuit', 'regulation', 'investigation',
            'patent', 'approval', 'launch', 'recall', 'insider', 'layoff', 'strike'
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a piece of text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment scores and metadata
        """
        if not text:
            return {'score': 0, 'magnitude': 0, 'classification': 'neutral'}
        
        # Basic preprocessing
        text = self._preprocess_text(text)
        
        # Get VADER sentiment
        vader_scores = self._get_vader_sentiment(text)
        
        # Get FinBERT sentiment if enabled
        if self.use_finbert:
            finbert_scores = self._get_finbert_sentiment(text)
        elif self.use_rule_based and RULE_BASED_FINBERT_AVAILABLE:
            finbert_scores = self._get_rule_based_sentiment(text)
        else:
            finbert_scores = None
        
        # Check for financial terms
        financial_terms_found = self._extract_financial_terms(text)
        
        # Apply rule-based adjustments
        adjusted_scores = self._apply_rule_based_adjustments(vader_scores, financial_terms_found)
        
        # Combine scores (weighted average if both available)
        final_scores = self._combine_scores(adjusted_scores, finbert_scores)
        
        # Add metadata and return
        result = {
            'text': text,  # Include the original text
            'score': final_scores['compound'],  # -1 to 1 range
            'magnitude': abs(final_scores['compound']),  # 0 to 1 range
            'positive': final_scores['pos'],
            'negative': final_scores['neg'],
            'neutral': final_scores['neu'],
            'classification': self._classify_sentiment(final_scores['compound']),
            'sentiment': 'positive' if final_scores['compound'] > 0 else ('negative' if final_scores['compound'] < 0 else 'neutral'),
            'financial_terms': list(financial_terms_found) if financial_terms_found else []
        }
        
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _get_vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get VADER sentiment scores.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary with VADER sentiment scores
        """
        if not self.vader:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
        
        try:
            return self.vader.polarity_scores(text)
        except Exception as e:
            self.logger.error(f"VADER analysis error: {str(e)}")
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
    
    def _get_finbert_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """
        Get FinBERT sentiment scores.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary with FinBERT sentiment scores or None if error
        """
        if not self.finbert or not self.tokenizer:
            return None
        
        try:
            # Truncate text if too long (FinBERT has token limit)
            if len(text) > 512:
                text = text[:512]
            
            # Tokenize and get sentiment
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Simple inference without JIT
            with torch.no_grad():
                outputs = self.finbert(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # FinBERT predicts positive, negative, neutral
            probs = probabilities[0].tolist()
            
            # Map to VADER-like format for easier integration
            compound = (probs[0] - probs[1])  # positive - negative, range -1 to 1
            
            return {
                'compound': compound,
                'pos': probs[0],
                'neg': probs[1],
                'neu': probs[2]
            }
        except Exception as e:
            self.logger.error(f"FinBERT analysis error: {str(e)}")
            return None
    
    def _get_rule_based_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """
        Get sentiment scores using the rule-based FinBERT implementation.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary with sentiment scores or None if error
        """
        if not RULE_BASED_FINBERT_AVAILABLE:
            return None
        
        try:
            # Use our rule-based implementation
            result = rule_based_finbert.analyze_sentiment(text)
            
            # Map to VADER-like format for easier integration
            sentiment = result['sentiment']
            score = result['normalized_score']
            
            # Convert to the same format as FinBERT for compatibility
            if sentiment == 'positive':
                pos, neg, neu = 0.7, 0.1, 0.2
            elif sentiment == 'negative':
                pos, neg, neu = 0.1, 0.7, 0.2
            else:  # neutral
                pos, neg, neu = 0.2, 0.2, 0.6
                
            # Adjust based on score intensity
            intensity = abs(score)
            if intensity > 0:
                if sentiment == 'positive':
                    pos = 0.5 + (intensity * 0.5)
                    neg = 0.5 - (intensity * 0.4)
                    neu = 1.0 - pos - neg
                elif sentiment == 'negative':
                    neg = 0.5 + (intensity * 0.5)
                    pos = 0.5 - (intensity * 0.4)
                    neu = 1.0 - pos - neg
            
            return {
                'compound': score,
                'pos': pos,
                'neg': neg,
                'neu': neu
            }
        except Exception as e:
            self.logger.error(f"Rule-based sentiment analysis error: {str(e)}")
            return None
    
    def _extract_financial_terms(self, text: str) -> set:
        """
        Extract financial terms from text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Set of financial terms found in text
        """
        # Find all financial terms in text
        found_terms = set()
        
        # Check for bullish terms
        for term in self.bullish_terms:
            if term in text:
                found_terms.add(('bullish', term))
        
        # Check for bearish terms
        for term in self.bearish_terms:
            if term in text:
                found_terms.add(('bearish', term))
        
        # Check for financial terms
        for term in self.financial_terms:
            if term in text:
                found_terms.add(('financial', term))
        
        return found_terms
    
    def _apply_rule_based_adjustments(self, scores: Dict[str, float], 
                                     financial_terms: set) -> Dict[str, float]:
        """
        Apply rule-based adjustments to sentiment scores.
        
        Args:
            scores: VADER sentiment scores
            financial_terms: Financial terms found in text
            
        Returns:
            Adjusted sentiment scores
        """
        # Make a copy of scores to avoid modifying original
        adjusted = scores.copy()
        
        # Count financial term types
        bullish_count = sum(1 for term_type, _ in financial_terms if term_type == 'bullish')
        bearish_count = sum(1 for term_type, _ in financial_terms if term_type == 'bearish')
        
        # Apply adjustments based on financial terms
        if bullish_count > bearish_count:
            # More bullish terms than bearish = amplify positive or reduce negative
            if adjusted['compound'] > 0:
                boost_factor = min(1.0, 0.1 * bullish_count)
                adjusted['compound'] = min(1.0, adjusted['compound'] * (1 + boost_factor))
            elif adjusted['compound'] < 0:
                reduction_factor = min(0.5, 0.05 * bullish_count)
                adjusted['compound'] = adjusted['compound'] * (1 - reduction_factor)
        
        elif bearish_count > bullish_count:
            # More bearish terms than bullish = amplify negative or reduce positive
            if adjusted['compound'] < 0:
                boost_factor = min(1.0, 0.1 * bearish_count)
                adjusted['compound'] = max(-1.0, adjusted['compound'] * (1 + boost_factor))
            elif adjusted['compound'] > 0:
                reduction_factor = min(0.5, 0.05 * bearish_count)
                adjusted['compound'] = adjusted['compound'] * (1 - reduction_factor)
        
        # Ensure scores stay in valid range
        adjusted['compound'] = max(-1.0, min(1.0, adjusted['compound']))
        
        return adjusted
    
    def _combine_scores(self, vader_scores: Dict[str, float], 
                        finbert_scores: Optional[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine scores from multiple models.
        
        Args:
            vader_scores: VADER sentiment scores
            finbert_scores: FinBERT sentiment scores (or None)
            
        Returns:
            Combined sentiment scores
        """
        if not finbert_scores:
            return vader_scores
        
        # Combine with weighted average (FinBERT more specialized for finance)
        combined = {}
        vader_weight = 0.4
        finbert_weight = 0.6
        
        combined['compound'] = (vader_scores['compound'] * vader_weight + 
                              finbert_scores['compound'] * finbert_weight)
        
        combined['pos'] = (vader_scores['pos'] * vader_weight + 
                         finbert_scores['pos'] * finbert_weight)
        
        combined['neg'] = (vader_scores['neg'] * vader_weight + 
                         finbert_scores['neg'] * finbert_weight)
        
        combined['neu'] = (vader_scores['neu'] * vader_weight + 
                         finbert_scores['neu'] * finbert_weight)
        
        return combined
    
    def _classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score.
        
        Args:
            compound_score: Compound sentiment score (-1 to 1)
            
        Returns:
            Sentiment classification string
        """
        if compound_score >= 0.05:
            return 'bullish'
        elif compound_score <= -0.05:
            return 'bearish'
        else:
            return 'neutral'
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze(text) for text in texts]
