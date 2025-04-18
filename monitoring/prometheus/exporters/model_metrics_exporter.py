"""
Model Metrics Exporter for Prometheus

This module collects and exports model-specific metrics including prediction
accuracy, latency, and feature importance for all ML models in the system.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Set

import numpy as np
import pandas as pd
from mlflow import MlflowClient

from .base_exporter import BaseExporter

logger = logging.getLogger(__name__)

class ModelMetricsExporter(BaseExporter):
    """
    Exporter for machine learning model metrics.
    
    Collects and exports detailed metrics for each model type:
    - Stock selection model
    - Entry timing model
    - Peak detection model
    - Risk sizing model
    - Market regime detection model
    """
    
    def __init__(self, model_interfaces=None, mlflow_uri=None, port: int = 8003, 
                interval: int = 30):
        """
        Initialize the model metrics exporter.
        
        Args:
            model_interfaces: Dictionary of model interfaces
            mlflow_uri: URI for MLflow tracking server
            port: Port to expose metrics on
            interval: Collection interval in seconds
        """
        super().__init__(name="model", port=port, interval=interval)
        
        # Store model interfaces
        self.model_interfaces = model_interfaces or {}
        
        # MLflow client for model registry metrics
        self.mlflow_client = MlflowClient(tracking_uri=mlflow_uri) if mlflow_uri else None
        
        # Model types to monitor
        self.model_types = [
            "stock_selection", "entry_timing", "peak_detection", 
            "risk_sizing", "market_regime"
        ]
        
        # Initialize metrics
        self._init_prediction_metrics()
        self._init_accuracy_metrics()
        self._init_feature_metrics()
        self._init_registry_metrics()
        
        logger.info("Initialized model metrics exporter")
    
    def _init_prediction_metrics(self) -> None:
        """Initialize prediction performance metrics."""
        # Prediction counters
        for model_type in self.model_types:
            self.create_counter(
                name=f"{model_type}_predictions_total",
                description=f"Total number of {model_type} model predictions"
            )
            
            self.create_counter(
                name=f"{model_type}_prediction_errors_total",
                description=f"Total number of {model_type} model prediction errors",
                labels=["error_type"]
            )
        
        # Prediction latency metrics
        for model_type in self.model_types:
            self.create_histogram(
                name=f"{model_type}_prediction_latency_ms",
                description=f"Latency of {model_type} model predictions in milliseconds",
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
            )
        
        # Batch size metrics
        for model_type in self.model_types:
            self.create_gauge(
                name=f"{model_type}_batch_size",
                description=f"Size of prediction batches for {model_type} model"
            )
        
        # Prediction cache metrics
        for model_type in self.model_types:
            self.create_gauge(
                name=f"{model_type}_cache_hit_rate",
                description=f"Cache hit rate for {model_type} model predictions"
            )
            
            self.create_gauge(
                name=f"{model_type}_cache_entries",
                description=f"Number of entries in {model_type} model prediction cache"
            )
    
    def _init_accuracy_metrics(self) -> None:
        """Initialize model accuracy metrics."""
        # Classification metrics
        for model_type in ["stock_selection", "entry_timing", "peak_detection", "market_regime"]:
            self.create_gauge(
                name=f"{model_type}_accuracy",
                description=f"Overall accuracy of {model_type} model"
            )
            
            self.create_gauge(
                name=f"{model_type}_precision",
                description=f"Precision of {model_type} model",
                labels=["class"]
            )
            
            self.create_gauge(
                name=f"{model_type}_recall",
                description=f"Recall of {model_type} model",
                labels=["class"]
            )
            
            self.create_gauge(
                name=f"{model_type}_f1_score",
                description=f"F1 score of {model_type} model",
                labels=["class"]
            )
            
            self.create_gauge(
                name=f"{model_type}_roc_auc",
                description=f"ROC AUC of {model_type} model",
                labels=["class"]
            )
        
        # Regression metrics (for risk_sizing model)
        self.create_gauge(
            name="risk_sizing_mse",
            description="Mean squared error of risk_sizing model"
        )
        
        self.create_gauge(
            name="risk_sizing_rmse",
            description="Root mean squared error of risk_sizing model"
        )
        
        self.create_gauge(
            name="risk_sizing_mae",
            description="Mean absolute error of risk_sizing model"
        )
        
        self.create_gauge(
            name="risk_sizing_r2",
            description="R-squared of risk_sizing model"
        )
        
        # Prediction distribution metrics
        for model_type in self.model_types:
            if model_type == "risk_sizing":
                self.create_gauge(
                    name="risk_sizing_prediction_mean",
                    description="Mean of risk_sizing model predictions"
                )
                
                self.create_gauge(
                    name="risk_sizing_prediction_std",
                    description="Standard deviation of risk_sizing model predictions"
                )
                
                self.create_gauge(
                    name="risk_sizing_prediction_min",
                    description="Minimum of risk_sizing model predictions"
                )
                
                self.create_gauge(
                    name="risk_sizing_prediction_max",
                    description="Maximum of risk_sizing model predictions"
                )
            else:
                self.create_gauge(
                    name=f"{model_type}_class_distribution",
                    description=f"Distribution of {model_type} model predictions",
                    labels=["class"]
                )
                
                self.create_gauge(
                    name=f"{model_type}_threshold",
                    description=f"Decision threshold for {model_type} model",
                    labels=["class"]
                )
        
        # Model drift metrics
        for model_type in self.model_types:
            self.create_gauge(
                name=f"{model_type}_drift_score",
                description=f"Data drift score for {model_type} model"
            )
            
            self.create_gauge(
                name=f"{model_type}_feature_drift",
                description=f"Feature drift for {model_type} model",
                labels=["feature"]
            )
    
    def _init_feature_metrics(self) -> None:
        """Initialize feature importance metrics."""
        for model_type in self.model_types:
            self.create_gauge(
                name=f"{model_type}_feature_importance",
                description=f"Feature importance for {model_type} model",
                labels=["feature"]
            )
            
            self.create_gauge(
                name=f"{model_type}_shap_value",
                description=f"SHAP value contribution for {model_type} model",
                labels=["feature"]
            )
    
    def _init_registry_metrics(self) -> None:
        """Initialize model registry metrics."""
        # Model version metrics
        self.create_gauge(
            name="model_version",
            description="Current version of models in use",
            labels=["model_type", "stage"]
        )
        
        self.create_gauge(
            name="model_age_days",
            description="Age of models in days",
            labels=["model_type", "stage"]
        )
        
        # Model performance metrics from registry
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "mse", "r2"]:
            self.create_gauge(
                name=f"registered_model_{metric}",
                description=f"Registered model {metric} from MLflow",
                labels=["model_type", "version", "stage"]
            )
        
        # Model artifact size
        self.create_gauge(
            name="model_artifact_size_mb",
            description="Size of model artifact in MB",
            labels=["model_type", "version"]
        )
    
    def collect(self) -> None:
        """Collect and update all model metrics."""
        try:
            # Collect model prediction metrics
            self._collect_prediction_metrics()
            
            # Collect model accuracy metrics
            self._collect_accuracy_metrics()
            
            # Collect feature importance metrics
            self._collect_feature_metrics()
            
            # Collect model registry metrics
            if self.mlflow_client:
                self._collect_registry_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting model metrics: {str(e)}")
    
    def _collect_prediction_metrics(self) -> None:
        """Collect model prediction performance metrics."""
        try:
            # For each model type
            for model_type, model in self.model_interfaces.items():
                if model_type not in self.model_types:
                    continue
                
                # Skip if model doesn't have stats method
                if not hasattr(model, "get_prediction_stats") or not callable(model.get_prediction_stats):
                    continue
                
                # Get prediction stats
                stats = model.get_prediction_stats()
                
                # Update batch size
                batch_size = stats.get("avg_batch_size", 0)
                self.get_metric(f"{model_type}_batch_size").set(batch_size)
                
                # Update cache metrics
                cache_hit_rate = stats.get("cache_hit_rate", 0)
                cache_entries = stats.get("cache_entries", 0)
                
                self.get_metric(f"{model_type}_cache_hit_rate").set(cache_hit_rate)
                self.get_metric(f"{model_type}_cache_entries").set(cache_entries)
                
                # Update recent latencies
                latencies = stats.get("recent_latencies_ms", [])
                latency_hist = self.get_metric(f"{model_type}_prediction_latency_ms")
                
                for latency in latencies:
                    latency_hist.observe(latency)
                
        except Exception as e:
            logger.error(f"Error collecting prediction metrics: {str(e)}")
    
    def _collect_accuracy_metrics(self) -> None:
        """Collect model accuracy metrics."""
        try:
            # For each model type
            for model_type, model in self.model_interfaces.items():
                if model_type not in self.model_types:
                    continue
                
                # Skip if model doesn't have stats method
                if not hasattr(model, "get_accuracy_stats") or not callable(model.get_accuracy_stats):
                    continue
                
                # Get accuracy stats
                stats = model.get_accuracy_stats()
                
                if model_type == "risk_sizing":
                    # Regression metrics for risk_sizing
                    self.get_metric("risk_sizing_mse").set(stats.get("mse", 0))
                    self.get_metric("risk_sizing_rmse").set(stats.get("rmse", 0))
                    self.get_metric("risk_sizing_mae").set(stats.get("mae", 0))
                    self.get_metric("risk_sizing_r2").set(stats.get("r2", 0))
                    
                    # Prediction distribution
                    pred_stats = stats.get("prediction_stats", {})
                    self.get_metric("risk_sizing_prediction_mean").set(pred_stats.get("mean", 0))
                    self.get_metric("risk_sizing_prediction_std").set(pred_stats.get("std", 0))
                    self.get_metric("risk_sizing_prediction_min").set(pred_stats.get("min", 0))
                    self.get_metric("risk_sizing_prediction_max").set(pred_stats.get("max", 0))
                    
                else:
                    # Classification metrics for other models
                    self.get_metric(f"{model_type}_accuracy").set(stats.get("accuracy", 0))
                    
                    # Per-class metrics
                    classes = stats.get("classes", [])
                    precision_metric = self.get_metric(f"{model_type}_precision")
                    recall_metric = self.get_metric(f"{model_type}_recall")
                    f1_metric = self.get_metric(f"{model_type}_f1_score")
                    roc_auc_metric = self.get_metric(f"{model_type}_roc_auc")
                    
                    for cls in classes:
                        class_stats = stats.get("per_class", {}).get(cls, {})
                        
                        precision_metric.labels(class_name=cls).set(class_stats.get("precision", 0))
                        recall_metric.labels(class_name=cls).set(class_stats.get("recall", 0))
                        f1_metric.labels(class_name=cls).set(class_stats.get("f1", 0))
                        roc_auc_metric.labels(class_name=cls).set(class_stats.get("roc_auc", 0))
                    
                    # Class distribution
                    dist_metric = self.get_metric(f"{model_type}_class_distribution")
                    distribution = stats.get("class_distribution", {})
                    
                    for cls, pct in distribution.items():
                        dist_metric.labels(class_name=cls).set(pct)
                    
                    # Thresholds
                    threshold_metric = self.get_metric(f"{model_type}_threshold")
                    thresholds = stats.get("thresholds", {})
                    
                    for cls, threshold in thresholds.items():
                        threshold_metric.labels(class_name=cls).set(threshold)
                
                # Drift metrics
                drift_score = stats.get("drift_score", 0)
                self.get_metric(f"{model_type}_drift_score").set(drift_score)
                
                feature_drift = stats.get("feature_drift", {})
                feature_drift_metric = self.get_metric(f"{model_type}_feature_drift")
                
                for feature, drift in feature_drift.items():
                    feature_drift_metric.labels(feature=feature).set(drift)
                
        except Exception as e:
            logger.error(f"Error collecting accuracy metrics: {str(e)}")
    
    def _collect_feature_metrics(self) -> None:
        """Collect feature importance metrics."""
        try:
            # For each model type
            for model_type, model in self.model_interfaces.items():
                if model_type not in self.model_types:
                    continue
                
                # Skip if model doesn't have feature importance method
                if not hasattr(model, "get_feature_importance") or not callable(model.get_feature_importance):
                    continue
                
                # Get feature importance
                importance = model.get_feature_importance()
                
                # Update feature importance metrics
                importance_metric = self.get_metric(f"{model_type}_feature_importance")
                
                for feature, value in importance.items():
                    importance_metric.labels(feature=feature).set(value)
                
                # Get SHAP values if available
                if hasattr(model, "get_shap_values") and callable(model.get_shap_values):
                    shap_values = model.get_shap_values()
                    shap_metric = self.get_metric(f"{model_type}_shap_value")
                    
                    for feature, value in shap_values.items():
                        shap_metric.labels(feature=feature).set(value)
                
        except Exception as e:
            logger.error(f"Error collecting feature metrics: {str(e)}")
    
    def _collect_registry_metrics(self) -> None:
        """Collect model registry metrics from MLflow."""
        try:
            # For each model type
            for model_type in self.model_types:
                try:
                    # Get model info from MLflow
                    model_name = f"trading_{model_type}"
                    model_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
                    
                    # Skip if no versions found
                    if not model_versions:
                        continue
                    
                    # Process each version
                    for version in model_versions:
                        version_num = version.version
                        stage = version.current_stage
                        
                        # Skip archived models
                        if stage == "Archived":
                            continue
                        
                        # Update version metric
                        self.get_metric("model_version").labels(
                            model_type=model_type, stage=stage
                        ).set(int(version_num))
                        
                        # Calculate model age
                        if version.creation_timestamp:
                            creation_time = int(version.creation_timestamp) / 1000  # Convert ms to seconds
                            age_days = (time.time() - creation_time) / (24 * 3600)
                            
                            self.get_metric("model_age_days").labels(
                                model_type=model_type, stage=stage
                            ).set(age_days)
                        
                        # Get run info
                        run_id = version.run_id
                        if run_id:
                            run = self.mlflow_client.get_run(run_id)
                            metrics = run.data.metrics
                            
                            # Update performance metrics
                            for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc", "mse", "r2"]:
                                if metric_name in metrics:
                                    self.get_metric(f"registered_model_{metric_name}").labels(
                                        model_type=model_type, version=version_num, stage=stage
                                    ).set(metrics[metric_name])
                            
                            # Get artifact info
                            artifacts = self.mlflow_client.list_artifacts(run_id, "model")
                            if artifacts:
                                total_size_mb = sum(a.file_size for a in artifacts if hasattr(a, "file_size")) / (1024 * 1024)
                                
                                self.get_metric("model_artifact_size_mb").labels(
                                    model_type=model_type, version=version_num
                                ).set(total_size_mb)
                    
                except Exception as e:
                    logger.error(f"Error collecting registry metrics for {model_type}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error collecting registry metrics: {str(e)}")
    
    # Helper methods to record metrics from outside the exporter
    
    def record_prediction(self, model_type: str, latency_ms: float) -> None:
        """
        Record a model prediction.
        
        Args:
            model_type: Type of model
            latency_ms: Prediction latency in milliseconds
        """
        if model_type not in self.model_types:
            return
        
        # Increment prediction counter
        self.get_metric(f"{model_type}_predictions_total").inc()
        
        # Record latency
        self.get_metric(f"{model_type}_prediction_latency_ms").observe(latency_ms)
    
    def record_prediction_error(self, model_type: str, error_type: str) -> None:
        """
        Record a model prediction error.
        
        Args:
            model_type: Type of model
            error_type: Type of error
        """
        if model_type not in self.model_types:
            return
        
        # Increment error counter
        self.get_metric(f"{model_type}_prediction_errors_total").labels(
            error_type=error_type
        ).inc()
    
    def record_execution(self, model_name: str, execution_time: float, success: bool = True, 
                      error: str = None, metadata: Dict[str, Any] = None) -> None:
        """
        Record a model execution.
        
        Args:
            model_name: Name of the model
            execution_time: Execution time in seconds
            success: Whether the execution was successful
            error: Error message if execution failed
            metadata: Additional metadata for the execution
        """
        if model_name not in self.model_types:
            # Try to map the model name to a model type
            model_type = model_name
            for known_type in self.model_types:
                if known_type in model_name:
                    model_type = known_type
                    break
        else:
            model_type = model_name
        
        # Record as prediction with latency
        self.record_prediction(model_type, execution_time * 1000)  # Convert to ms
        
        # Record error if execution failed
        if not success and error:
            self.record_prediction_error(model_type, error[:50])  # Truncate long errors
    
    def get_model_metrics(self, name: str) -> Dict[str, Any]:
        """
        Get metrics for a specific model.
        
        Args:
            name: Name of the model
            
        Returns:
            Dictionary with model metrics
        """
        # This is a placeholder implementation
        # In a real implementation, this would return actual metrics from Prometheus
        return {
            "predictions_total": 0,
            "prediction_errors_total": 0,
            "avg_latency_ms": 0,
            "cache_hit_rate": 0,
            "accuracy": 0
        }
