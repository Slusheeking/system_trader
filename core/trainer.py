#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core Abstract Trainer for Day Trading System
-------------------------------------------
Defines BaseTrainer with a standardized MLflow-tracked training pipeline.
"""

import abc
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow.tracking import get_tracker
from models.explainability import ModelExplainer

# Setup logging
logger = logging.getLogger(__name__)


class BaseTrainer(abc.ABC):
    """
    Abstract base trainer providing a template method `run` for standard
    data loading, feature preparation, model training, evaluation, logging,
    and saving using MLflow.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the BaseTrainer.

        Args:
            params: Hyperparameters or config values to log
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        self.params: Dict[str, Any] = params or {}
        self.tracker = get_tracker(
            experiment_name or self.__class__.__name__,
            model_version="1.0",
            tracking_uri=tracking_uri
        )
        self.explainer: Optional[ModelExplainer] = None
        self.metrics = {}
        
        # Create results directory if specified in params
        self.results_dir = self.params.get('results_dir', 'models')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"{self.__class__.__name__} initialized")

    @abc.abstractmethod
    def load_data(self) -> Any:
        """
        Load and return raw data required for training.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_features(self, data: Any) -> Any:
        """
        Transform raw data into feature set for model training.

        Args:
            data: Output from load_data

        Returns:
            Processed features
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_features(self, features: Any, target: Any) -> Any:
        """
        Select and return a subset of features for model training.

        Args:
            features: Processed feature set
            target: Target values

        Returns:
            Selected features
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_target(self, data: Any) -> Any:
        """
        Generate and return target values/labels from raw data.

        Args:
            data: Output from load_data

        Returns:
            Target values for training
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_model(self, features: Any, target: Any) -> Any:
        """
        Train and return the model.

        Args:
            features: Processed feature set
            target: Target values

        Returns:
            Trained model object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, model: Any, features: Any, target: Any) -> Dict[str, float]:
        """
        Evaluate the trained model and return performance metrics.

        Args:
            model: Trained model
            features: Feature set used for evaluation
            target: True target values

        Returns:
            Dictionary of metric names and values
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, model: Any) -> None:
        """
        Persist the trained model to storage.

        Args:
            model: Trained model
        """
        raise NotImplementedError

    def run(self) -> None:
        """
        Execute the end-to-end training pipeline:
        1. Start MLflow run
        2. Log hyperparameters
        3. Load data
        4. Prepare features
        5. Generate target
        6. Select features
        7. Train model
        8. Evaluate and log metrics
        9. Generate model explainability
        10. Log trained model
        11. Save model to disk
        12. End MLflow run
        """
        # Begin tracking
        run_name = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracker.start_run(run_name=run_name)

        # Log any provided parameters
        if self.params:
            self.tracker.log_params(self.params)

        try:
            # Pipeline steps
            logger.info("Loading data")
            data = self.load_data()
            
            logger.info("Preparing features")
            features = self.prepare_features(data)
            
            logger.info("Generating target")
            target = self.generate_target(data)
            
            logger.info("Selecting features")
            selected_features = self.select_features(features, target)
            
            logger.info("Training model")
            model = self.train_model(selected_features, target)

            # Evaluation
            logger.info("Evaluating model")
            metrics = self.evaluate(model, features, target)
            self.metrics = metrics
            self.tracker.log_metrics(metrics)

            # Model Explainability
            logger.info("Generating model explainability")
            self._analyze_model_explainability(features, model)

            # Model artifact
            logger.info("Logging model to MLflow")
            self.tracker.log_model(model, artifact_path="model")

            # Persist model locally
            logger.info("Saving model to disk")
            self.save_model(model)
            
            logger.info("Training pipeline completed successfully")
        except Exception as e:
            # Log any exceptions that occur during the pipeline
            self.tracker.log_text(str(e), "pipeline_error.txt")
            logger.error(f"Error during training pipeline: {e}")
            raise  # Re-raise the exception to indicate failure
        finally:
            # Ensure the MLflow run is always closed
            self.tracker.end_run()

    def get_target_column_name(self) -> str:
        """
        Abstract method to get the name of the target column.
        This should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def _analyze_model_explainability(self, features: Any, model: Any) -> None:
        """
        Analyze model explainability and generate visualizations.
        
        Args:
            features: Feature data used for analysis
            model: Trained model to explain
        """
        try:
            # Create explainer if it doesn't exist
            if self.explainer is None:
                output_dir = os.path.join(self.results_dir, "explainability")
                os.makedirs(output_dir, exist_ok=True)
                self.explainer = ModelExplainer(model=model, output_dir=output_dir)
            
            # Get target column name
            target_col = self.get_target_column_name()
            
            # Generate explanations
            # Extract features and target, handling DataFrames and other data types
            if isinstance(features, pd.DataFrame) and target_col in features.columns:
                X = features.drop(target_col, axis=1, errors='ignore')
                y = features.get(target_col)
            else:
                # If target isn't in features or features isn't a DataFrame, use as is
                X = features
                y = None
            
            explainability_results = self.explainer.explain_model(X=X, y=y)
            
            # Log explainability artifacts
            if 'shap' in explainability_results and 'summary_plot_path' in explainability_results['shap']:
                self.tracker.log_artifact(explainability_results['shap']['summary_plot_path'])
            
            if 'feature_importance' in explainability_results:
                if 'builtin_plot_path' in explainability_results['feature_importance']:
                    self.tracker.log_artifact(explainability_results['feature_importance']['builtin_plot_path'])
                if 'permutation_plot_path' in explainability_results['feature_importance']:
                    self.tracker.log_artifact(explainability_results['feature_importance']['permutation_plot_path'])
            
            if 'feature_interactions' in explainability_results and 'interaction_plot_path' in explainability_results['feature_interactions']:
                self.tracker.log_artifact(explainability_results['feature_interactions']['interaction_plot_path'])
            
            if 'partial_dependence' in explainability_results:
                if 'pdp_paths' in explainability_results['partial_dependence']:
                    for plot_path in explainability_results['partial_dependence']['pdp_paths']:
                        self.tracker.log_artifact(plot_path)
                if 'pdp_2d_paths' in explainability_results['partial_dependence']:
                    for plot_path in explainability_results['partial_dependence']['pdp_2d_paths']:
                        self.tracker.log_artifact(plot_path)
            
        except Exception as e:
            self.tracker.log_text(str(e), "explainability_error.txt")
            logger.error(f"Error during model explainability: {e}")
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
