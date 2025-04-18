"""
MLflow scikit-learn module for Day Trading System

This module provides MLflow integration for scikit-learn models.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def log_model(model, artifact_path, conda_env=None, signature=None, 
             input_example=None, registered_model_name=None):
    """
    Log a scikit-learn model as an MLflow artifact.
    
    Args:
        model: scikit-learn model to log
        artifact_path: Path within the MLflow run artifacts
        conda_env: Conda environment for the model
        signature: Model signature (inputs and outputs)
        input_example: Example of model inputs
        registered_model_name: If provided, model will be registered with this name
    """
    import mlflow
    mlflow.log_model(model, artifact_path)
    logger.info(f"Logged scikit-learn model to {artifact_path}")
