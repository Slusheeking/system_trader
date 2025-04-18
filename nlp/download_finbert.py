#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FinBERT Model Downloader
-----------------------
This script downloads the FinBERT model from Hugging Face
and saves it locally for offline use.
"""

import os
import sys
import argparse
from typing import Optional

def download_finbert(output_dir: Optional[str] = None) -> str:
    """
    Download the FinBERT model from Hugging Face.
    
    Args:
        output_dir: Directory to save the model to. If None, uses './finbert_model'
        
    Returns:
        Path to the downloaded model
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("Error: transformers library not installed.")
        print("Please install it with: pip install transformers")
        sys.exit(1)
        
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed.")
        print("Please install it with: pip install torch")
        sys.exit(1)
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "finbert_model")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading FinBERT model to {output_dir}...")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer downloaded successfully.")
        
        # Download model
        print("Downloading model (this may take a while)...")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.save_pretrained(output_dir)
        print("Model downloaded successfully.")
        
        print(f"\nFinBERT model and tokenizer have been downloaded to: {output_dir}")
        print("\nYou can now use this model with the sentiment analyzer by setting:")
        print("1. In code: SentimentAnalyzer(use_finbert=True, model_path='path/to/finbert_model')")
        print("2. In config: finbert_path: 'path/to/finbert_model'")
        
        return output_dir
    
    except Exception as e:
        print(f"Error downloading FinBERT model: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Download FinBERT model for offline use')
    parser.add_argument('--output-dir', type=str, help='Directory to save the model to')
    
    args = parser.parse_args()
    
    # Check if NumPy compatibility issues might occur
    try:
        import numpy as np
        version = np.__version__
        major_version = int(version.split('.')[0])
        if major_version >= 2:
            print(f"Warning: NumPy version {version} detected.")
            print("This may cause compatibility issues with the transformers library.")
            print("If you encounter errors, consider using the rule-based implementation instead.")
            print("Continue anyway? (y/n)")
            response = input().lower()
            if response != 'y':
                print("Download cancelled.")
                sys.exit(0)
    except ImportError:
        pass
    
    # Download the model
    download_finbert(args.output_dir)

if __name__ == "__main__":
    print("=== FinBERT Model Downloader ===")
    main()