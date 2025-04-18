"""
MLflow TensorFlow module for Day Trading System

This module provides MLflow integration for TensorFlow models.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def log_model(model, artifact_path, conda_env=None, signature=None, 
             input_example=None, registered_model_name=None):
    """
    Log a TensorFlow model as an MLflow artifact.
    
    Args:
        model: TensorFlow model to log
        artifact_path: Path within the MLflow run artifacts
        conda_env: Conda environment for the model
        signature: Model signature (inputs and outputs)
        input_example: Example of model inputs
        registered_model_name: If provided, model will be registered with this name
    """
    import mlflow
    mlflow.log_model(model, artifact_path)
    logger.info(f"Logged TensorFlow model to {artifact_path}")
