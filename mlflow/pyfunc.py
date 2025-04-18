"""
MLflow PyFunc module for Day Trading System

This module provides MLflow integration for Python functions and custom models.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def log_model(artifact_path, python_model, conda_env=None, signature=None, 
             input_example=None, registered_model_name=None):
    """
    Log a Python function or custom model as an MLflow artifact.
    
    Args:
        artifact_path: Path within the MLflow run artifacts
        python_model: Python model to log
        conda_env: Conda environment for the model
        signature: Model signature (inputs and outputs)
        input_example: Example of model inputs
        registered_model_name: If provided, model will be registered with this name
    """
    import mlflow
    mlflow.log_model(python_model, artifact_path)
    logger.info(f"Logged Python function model to {artifact_path}")
