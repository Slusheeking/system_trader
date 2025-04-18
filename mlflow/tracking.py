"""
MLflow Tracking Module for Day Trading System

This module provides a robust experiment tracking and logging functionality using MLflow.
It enables tracking model experiments, parameters, metrics, and artifacts for
reproducibility and comparison of different model versions.
"""

import os
import yaml
import mlflow
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowTracker:
    """
    A wrapper around MLflow for experiment tracking in the day trading system.
    Handles experiment creation, run management, and metrics/artifact logging.
    """
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = None, registry_uri: str = None, experiment_prefix: str = None):
        """
        Initialize the MLflow tracking wrapper.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
            registry_uri: MLflow model registry URI (optional)
            experiment_prefix: Prefix for experiment names (optional)
        """
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Using MLflow tracking URI: {tracking_uri}")
        else:
            # Try to get tracking URI, fall back to a default message if not available
            try:
                tracking_uri = mlflow.get_tracking_uri()
                logger.info(f"Using default MLflow tracking URI: {tracking_uri}")
            except AttributeError:
                logger.info("Using default MLflow tracking URI")
                
        # Set registry URI if provided
        if registry_uri:
            try:
                mlflow.set_registry_uri(registry_uri)
                logger.info(f"Using MLflow registry URI: {registry_uri}")
            except (AttributeError, ImportError) as e:
                logger.warning(f"Failed to set MLflow registry URI: {str(e)}")
        
        # Set experiment
        self.experiment_prefix = experiment_prefix or ""
        self.experiment_name = experiment_name
        if self.experiment_name is None:
            self.experiment_name = f"{self.experiment_prefix}day_trading_default" if self.experiment_prefix else "day_trading_default"
        elif self.experiment_prefix and not self.experiment_name.startswith(self.experiment_prefix):
            self.experiment_name = f"{self.experiment_prefix}{self.experiment_name}"
        self._setup_experiment()
        
        # Active run reference
        self.active_run = None
    
    def _setup_experiment(self):
        """Create or get the experiment."""
        try:
            # Try to use the installed MLflow library
            import mlflow as mlflow_lib
            experiment = mlflow_lib.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow_lib.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {self.experiment_id})")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {self.experiment_id})")
        except Exception as e:
            # Fallback to a default experiment ID if MLflow is not available
            logger.warning(f"Error setting up MLflow experiment: {str(e)}")
            logger.warning("Using default experiment ID")
            self.experiment_id = "0"  # Default experiment ID
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False,
                 tags: Optional[Dict[str, Any]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            nested: Whether this is a nested run
            tags: Dictionary of tags to attach to the run
            
        Returns:
            An active MLflow run
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add model type and version to tags if not provided
        if tags is None:
            tags = {}
        
        try:
            # Try to use the installed MLflow library
            import mlflow as mlflow_lib
            self.active_run = mlflow_lib.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags
            )
            logger.info(f"Started MLflow run: {run_name} (ID: {self.active_run.info.run_id})")
        except Exception as e:
            # Create a dummy run object if MLflow is not available
            logger.warning(f"Error starting MLflow run: {str(e)}")
            logger.warning("Creating dummy run")
            
            # Create a simple object to mimic MLflow's RunInfo
            class DummyRunInfo:
                def __init__(self):
                    self.run_id = f"dummy_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Create a simple object to mimic MLflow's ActiveRun
            class DummyActiveRun:
                def __init__(self, run_name):
                    self.info = DummyRunInfo()
                    self.data = None
                    self.run_name = run_name
            
            self.active_run = DummyActiveRun(run_name)
            logger.info(f"Started dummy run: {run_name} (ID: {self.active_run.info.run_id})")
        
        return self.active_run
    
    def end_run(self):
        """End the current active run."""
        if self.active_run:
            try:
                # Try to use the installed MLflow library
                import mlflow as mlflow_lib
                mlflow_lib.end_run()
                logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
            except Exception as e:
                # Just log a message if MLflow is not available
                logger.warning(f"Error ending MLflow run: {str(e)}")
                logger.info(f"Ended dummy run: {self.active_run.info.run_id}")
            
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run to log parameters.")
            self.start_run()
        
        try:
            # Try to use the installed MLflow library
            import mlflow as mlflow_lib
            mlflow_lib.log_params(params)
            logger.debug(f"Logged {len(params)} parameters to run {self.active_run.info.run_id}")
        except Exception as e:
            # Just log a message if MLflow is not available
            logger.warning(f"Error logging parameters with MLflow: {str(e)}")
            logger.info(f"Would have logged {len(params)} parameters to run {self.active_run.info.run_id}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step value for the metrics
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run to log metrics.")
            self.start_run()
        
        try:
            # Try to use the installed MLflow library
            import mlflow as mlflow_lib
            mlflow_lib.log_metrics(metrics, step=step)
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.debug(f"Logged metrics at step {step}: {metrics_str}")
        except Exception as e:
            # Just log a message if MLflow is not available
            logger.warning(f"Error logging metrics with MLflow: {str(e)}")
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Would have logged metrics at step {step}: {metrics_str}")
    
    def log_artifact(self, local_path: str):
        """
        Log a local file or directory as an artifact.
        
        Args:
            local_path: Path to the local file or directory
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run to log artifact.")
            self.start_run()
            
        mlflow.log_artifact(local_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_figure(self, figure, artifact_path: str):
        """
        Log a matplotlib or plotly figure.
        
        Args:
            figure: The figure object to log
            artifact_path: The path within the artifact directory
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run to log figure.")
            self.start_run()
            
        mlflow.log_figure(figure, artifact_path)
        logger.debug(f"Logged figure as: {artifact_path}")
    
    def log_model(self, model, artifact_path: str, conda_env=None, signature=None, 
                  input_example=None, registered_model_name=None):
        """
        Log a model to the current run.
        
        Args:
            model: Model object to log
            artifact_path: Path within the MLflow run artifacts
            conda_env: Conda environment for the model
            signature: Model signature (inputs and outputs)
            input_example: Example of model inputs
            registered_model_name: If provided, model will be registered with this name
        """
        if not self.active_run:
            logger.warning("No active run. Starting a new run to log model.")
            self.start_run()
            
        # Determine model flavor based on model type
        if hasattr(model, "__module__") and "xgboost" in model.__module__:
            mlflow.xgboost.log_model(
                model, 
                artifact_path=artifact_path, 
                conda_env=conda_env,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif hasattr(model, "__module__") and "sklearn" in model.__module__:
            mlflow.sklearn.log_model(
                model, 
                artifact_path=artifact_path, 
                conda_env=conda_env,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif hasattr(model, "__module__") and "tensorflow" in model.__module__:
            mlflow.tensorflow.log_model(
                model, 
                artifact_path=artifact_path, 
                conda_env=conda_env,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        else:
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                conda_env=conda_env,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
            
        logger.info(f"Logged model to {artifact_path}" + 
                   (f" and registered as {registered_model_name}" if registered_model_name else ""))
    
    def log_feature_importance(self, model, feature_names: List[str], output_path: str = "feature_importance.png"):
        """
        Log feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            output_path: Path to save the feature importance plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
            else:
                logger.warning("Model doesn't have feature_importances_ or coef_ attribute.")
                return
                
            # Create DataFrame of feature importances
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importances
            plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
            sns.barplot(x='Importance', y='Feature', data=fi_df)
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save and log the figure
            plt.savefig(output_path)
            self.log_artifact(output_path)
            
            # Also log as CSV
            csv_path = output_path.replace('.png', '.csv')
            fi_df.to_csv(csv_path, index=False)
            self.log_artifact(csv_path)
            
            # Clean up local files
            plt.close()
            
        except Exception as e:
            logger.error(f"Error logging feature importance: {str(e)}")
    
    def log_confusion_matrix(self, y_true, y_pred, output_path: str = "confusion_matrix.png"):
        """
        Log confusion matrix for classification models.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            output_path: Path to save the confusion matrix plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Save and log the figure
            plt.savefig(output_path)
            self.log_artifact(output_path)
            
            # Clean up local files
            plt.close()
            
        except Exception as e:
            logger.error(f"Error logging confusion matrix: {str(e)}")
    
    def log_performance_metrics(self, y_true, y_pred, y_prob=None, 
                             classification: bool = True):
        """
        Log comprehensive performance metrics.
        
        Args:
            y_true: Ground truth values/labels
            y_pred: Predicted values/labels
            y_prob: Predicted probabilities (for classification)
            classification: Whether this is a classification task
        """
        if classification:
            self._log_classification_metrics(y_true, y_pred, y_prob)
        else:
            self._log_regression_metrics(y_true, y_pred)
    
    def _log_classification_metrics(self, y_true, y_pred, y_prob=None):
        """Log metrics for classification models."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add probability-based metrics if available
        if y_prob is not None:
            try:
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    # For multi-class, use the probability of the predicted class
                    if y_true.ndim == 1:
                        from sklearn.preprocessing import label_binarize
                        classes = np.unique(y_true)
                        y_true_bin = label_binarize(y_true, classes=classes)
                        metrics["roc_auc"] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
                        metrics["avg_precision"] = average_precision_score(y_true_bin, y_prob, average='weighted')
                else:
                    # For binary classification
                    pos_prob = y_prob if len(y_prob.shape) == 1 else y_prob[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_true, pos_prob)
                    metrics["avg_precision"] = average_precision_score(y_true, pos_prob)
            except Exception as e:
                logger.warning(f"Could not compute probability-based metrics: {str(e)}")
        
        # Log the metrics
        self.log_metrics(metrics)
        
        # If possible, create and log ROC curve
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                import matplotlib.pyplot as plt
                from sklearn.metrics import roc_curve, auc
                
                pos_prob = y_prob if len(y_prob.shape) == 1 else y_prob[:, 1]
                fpr, tpr, _ = roc_curve(y_true, pos_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                
                # Save and log the figure
                roc_path = "roc_curve.png"
                plt.savefig(roc_path)
                self.log_artifact(roc_path)
                plt.close()
                
            except Exception as e:
                logger.warning(f"Could not create ROC curve: {str(e)}")
    
    def _log_regression_metrics(self, y_true, y_pred):
        """Log metrics for regression models."""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            explained_variance_score, mean_absolute_percentage_error
        )
        
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),
        }
        
        # Add MAPE if there are no zeros in y_true
        if not np.any(y_true == 0):
            metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
        
        # Log the metrics
        self.log_metrics(metrics)
        
        # Create and log residual plot
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            residuals = y_true - y_pred
            
            # Create residual plot
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            
            # Save and log the figure
            residual_path = "residual_plot.png"
            plt.savefig(residual_path)
            self.log_artifact(residual_path)
            plt.close()
            
            # Create actual vs predicted plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r-')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted')
            
            # Save and log the figure
            pred_vs_actual_path = "actual_vs_predicted.png"
            plt.savefig(pred_vs_actual_path)
            self.log_artifact(pred_vs_actual_path)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create regression plots: {str(e)}")
    
    def log_trading_performance(self, 
                              trades_df: pd.DataFrame, 
                              portfolio_history: pd.DataFrame = None,
                              benchmark_history: pd.DataFrame = None,
                              model_name: str = "model"):
        """
        Log trading performance metrics and visualizations.
        
        Args:
            trades_df: DataFrame with trade details
            portfolio_history: DataFrame with portfolio value over time
            benchmark_history: DataFrame with benchmark value over time
            model_name: Name of the model for reference
        """
        # Check for active run
        if not self.active_run:
            logger.warning("No active run. Starting a new run to log trading performance.")
            self.start_run(run_name=f"{model_name}_trading_performance")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.dates import DateFormatter
            
            # Calculate trade metrics
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] <= 0]
            
            metrics = {
                "total_trades": len(trades_df),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                "total_profit": trades_df['profit'].sum(),
                "avg_profit_per_trade": trades_df['profit'].mean(),
                "avg_win": winning_trades['profit'].mean() if len(winning_trades) > 0 else 0,
                "avg_loss": losing_trades['profit'].mean() if len(losing_trades) > 0 else 0,
                "max_win": winning_trades['profit'].max() if len(winning_trades) > 0 else 0,
                "max_loss": losing_trades['profit'].min() if len(losing_trades) > 0 else 0,
                "profit_factor": abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) 
                               if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else float('inf')
            }
            
            # Log trade metrics
            self.log_metrics(metrics)
            
            # Create and save trade distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(trades_df['profit'], kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.xlabel('Profit/Loss')
            plt.ylabel('Frequency')
            plt.title('Trade Profit/Loss Distribution')
            
            trade_dist_path = f"{model_name}_trade_distribution.png"
            plt.savefig(trade_dist_path)
            self.log_artifact(trade_dist_path)
            plt.close()
            
            # If portfolio history is available, create equity curve
            if portfolio_history is not None and not portfolio_history.empty:
                plt.figure(figsize=(12, 6))
                
                # Plot portfolio value
                plt.plot(portfolio_history.index, portfolio_history['value'], 
                        label='Portfolio Value', linewidth=2)
                
                # Plot benchmark if available
                if benchmark_history is not None and not benchmark_history.empty:
                    plt.plot(benchmark_history.index, benchmark_history['value'], 
                            label='Benchmark', linewidth=2, alpha=0.7)
                
                # Format plot
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.title('Equity Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Format date axis
                date_format = DateFormatter('%Y-%m-%d')
                plt.gca().xaxis.set_major_formatter(date_format)
                plt.xticks(rotation=45)
                
                equity_curve_path = f"{model_name}_equity_curve.png"
                plt.savefig(equity_curve_path)
                self.log_artifact(equity_curve_path)
                plt.close()
                
                # Calculate and log additional portfolio metrics
                if len(portfolio_history) > 1:
                    returns = portfolio_history['value'].pct_change().dropna()
                    
                    # Benchmark returns if available
                    benchmark_returns = None
                    if benchmark_history is not None and not benchmark_history.empty:
                        benchmark_returns = benchmark_history['value'].pct_change().dropna()
                    
                    portfolio_metrics = self._calculate_portfolio_metrics(returns, benchmark_returns)
                    self.log_metrics(portfolio_metrics)
            
        except Exception as e:
            logger.error(f"Error logging trading performance: {str(e)}")
    
    def _calculate_portfolio_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Dictionary of portfolio metrics
        """
        import numpy as np
        
        # Assume returns are daily by default
        periods_per_year = 252
        
        metrics = {}
        
        # Total return
        total_return = (1 + returns).prod() - 1
        metrics["total_return"] = total_return
        
        # Annualized return
        num_periods = len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
        metrics["annualized_return"] = annualized_return
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(periods_per_year)
        metrics["annualized_volatility"] = volatility
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        metrics["sharpe_ratio"] = sharpe_ratio
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        metrics["max_drawdown"] = max_drawdown
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        metrics["calmar_ratio"] = calmar_ratio
        
        # Benchmark comparison metrics
        if benchmark_returns is not None:
            # Align returns
            combined = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(combined) > 0:
                port_returns = combined.iloc[:, 0]
                bench_returns = combined.iloc[:, 1]
                
                # Beta
                covariance = port_returns.cov(bench_returns)
                benchmark_variance = bench_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                metrics["beta"] = beta
                
                # Alpha (annualized)
                port_annualized_return = (1 + port_returns.sum()) ** (periods_per_year / len(port_returns)) - 1
                bench_annualized_return = (1 + bench_returns.sum()) ** (periods_per_year / len(bench_returns)) - 1
                alpha = port_annualized_return - (beta * bench_annualized_return)
                metrics["alpha"] = alpha
                
                # Information ratio
                tracking_error = (port_returns - bench_returns).std() * np.sqrt(periods_per_year)
                information_ratio = (port_annualized_return - bench_annualized_return) / tracking_error if tracking_error != 0 else 0
                metrics["information_ratio"] = information_ratio
        
        return metrics
    
    @staticmethod
    def get_run_by_id(run_id: str):
        """
        Get a run by ID.
        
        Args:
            run_id: The MLflow run ID
            
        Returns:
            MLflow run object
        """
        return mlflow.get_run(run_id)
    
    @staticmethod
    def search_runs(experiment_ids=None, filter_string="", order_by=None, max_results=1000):
        """
        Search for runs in the specified experiments.
        
        Args:
            experiment_ids: List of experiment IDs to search
            filter_string: Filter string for the search
            order_by: Column to order by
            max_results: Maximum number of results
            
        Returns:
            DataFrame of runs
        """
        return mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results
        )

# Convenience function to get a tracker instance with common configuration
def get_tracker(model_type: str, model_version: str = None, tracking_uri: str = None,
               registry_uri: str = None, experiment_prefix: str = None):
    """
    Get a configured MLflowTracker instance for a specific model type.
    
    Args:
        model_type: Type of model (e.g., 'stock_selection', 'entry_timing')
        model_version: Version identifier for the model
        tracking_uri: MLflow tracking server URI
        registry_uri: MLflow model registry URI
        experiment_prefix: Prefix for experiment names (optional)
        
    Returns:
        Configured MLflowTracker instance
    """
    # Create experiment name from model type
    experiment_name = f"{model_type}"
    
    # Create tags dictionary
    tags = {
        "model_type": model_type,
        "model_version": model_version or "development",
        "system": "day_trading"
    }
    
    # Create and configure tracker
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registry_uri=registry_uri,
        experiment_prefix=experiment_prefix
    )
    
    # Start run with tags
    tracker.start_run(tags=tags)
    
    return tracker
