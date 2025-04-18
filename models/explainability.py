#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Explainability Module
---------------------------
This module provides comprehensive model explainability tools that can be used
across all model types in the system. It includes SHAP value integration,
feature interaction analysis, and partial dependence plots.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Import explainability libraries
import shap
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Setup logging
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Provides model explainability tools for all model types.
    
    This class handles:
    1. SHAP value calculation and visualization
    2. Feature interaction analysis
    3. Partial dependence plots
    4. Permutation importance
    5. Global and local explanations
    """
    
    def __init__(self, model: Any, model_type: str = None, output_dir: str = None):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained model object
            model_type: Type of model ('classifier', 'regressor', 'deep_learning', etc.)
            output_dir: Directory to save explainability artifacts
        """
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type is None else model_type
        self.output_dir = output_dir or 'explainability_output'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize explainers
        self.shap_explainer = None
        self.is_tree_model = self._is_tree_based_model(model)
        
        logger.info(f"ModelExplainer initialized for {self.model_type} model")
    
    def _detect_model_type(self, model: Any) -> str:
        """
        Detect the type of model based on its class.
        
        Args:
            model: Trained model object
            
        Returns:
            String indicating model type
        """
        model_class = model.__class__.__name__
        
        if hasattr(model, "predict_proba"):
            return "classifier"
        elif hasattr(model, "predict"):
            return "regressor"
        elif "keras" in str(type(model)) or "tensorflow" in str(type(model)):
            return "deep_learning"
        elif "xgboost" in str(type(model)):
            return "xgboost"
        elif "lightgbm" in str(type(model)):
            return "lightgbm"
        elif "catboost" in str(type(model)):
            return "catboost"
        else:
            logger.warning(f"Unknown model type: {model_class}, defaulting to 'unknown'")
            return "unknown"
    
    def _is_tree_based_model(self, model: Any) -> bool:
        """
        Check if the model is tree-based.
        
        Args:
            model: Trained model object
            
        Returns:
            Boolean indicating if model is tree-based
        """
        tree_based_models = [
            "RandomForest", "XGBoost", "LightGBM", "CatBoost", 
            "DecisionTree", "ExtraTrees", "GradientBoosting"
        ]
        
        model_class = model.__class__.__name__
        return any(tree_type in model_class for tree_type in tree_based_models)
    
    def _initialize_shap_explainer(self, X: pd.DataFrame):
        """
        Initialize the appropriate SHAP explainer based on model type.
        
        Args:
            X: Feature data to use for explainer initialization
        """
        try:
            if self.is_tree_model:
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized SHAP TreeExplainer")
            elif self.model_type == "deep_learning":
                # For deep learning models, use DeepExplainer or GradientExplainer
                try:
                    self.shap_explainer = shap.DeepExplainer(self.model, X.values)
                    logger.info("Initialized SHAP DeepExplainer")
                except Exception:
                    self.shap_explainer = shap.GradientExplainer(self.model, X.values)
                    logger.info("Initialized SHAP GradientExplainer")
            else:
                # For other models, use KernelExplainer
                self.shap_explainer = shap.KernelExplainer(self.model.predict, X)
                logger.info("Initialized SHAP KernelExplainer")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            self.shap_explainer = None
    
    def explain_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                     feature_names: Optional[List[str]] = None,
                     sample_size: int = 1000) -> Dict[str, Any]:
        """
        Generate comprehensive model explanations.
        
        Args:
            X: Feature data
            y: Target data (optional)
            feature_names: List of feature names (optional)
            sample_size: Number of samples to use for explanations
            
        Returns:
            Dictionary with explanation results
        """
        logger.info("Generating comprehensive model explanations")
        
        # Sample data if needed
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
            if y is not None:
                y_sample = y.loc[X_sample.index]
            else:
                y_sample = None
        else:
            X_sample = X
            y_sample = y
        
        # Use column names if feature_names not provided
        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        
        # Initialize results dictionary
        results = {}
        
        # Generate SHAP explanations
        shap_results = self.generate_shap_explanations(X_sample, feature_names)
        results['shap'] = shap_results
        
        # Generate feature importance
        importance_results = self.calculate_feature_importance(X_sample, y_sample, feature_names)
        results['feature_importance'] = importance_results
        
        # Generate feature interactions
        interaction_results = self.analyze_feature_interactions(X_sample, feature_names)
        results['feature_interactions'] = interaction_results
        
        # Generate partial dependence plots
        pdp_results = self.generate_partial_dependence_plots(X_sample, feature_names)
        results['partial_dependence'] = pdp_results
        
        return results
    
    def generate_shap_explanations(self, X: pd.DataFrame, 
                                 feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate SHAP-based explanations for the model.
        
        Args:
            X: Feature data
            feature_names: List of feature names
            
        Returns:
            Dictionary with SHAP explanation results
        """
        logger.info("Generating SHAP explanations")
        
        results = {}
        
        # Initialize SHAP explainer if not already done
        if self.shap_explainer is None:
            self._initialize_shap_explainer(X)
        
        if self.shap_explainer is None:
            logger.error("SHAP explainer initialization failed")
            return results
        
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X)
            results['shap_values'] = shap_values
            
            # Handle multi-class classification case
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # For multi-class, calculate mean absolute SHAP values across classes
                mean_abs_shap = np.abs(np.array(shap_values)).mean(0)
                results['mean_abs_shap'] = mean_abs_shap
                
                # Feature importance based on mean absolute SHAP values
                if feature_names is not None:
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': np.abs(np.array(shap_values)).mean(0).mean(0)
                    }).sort_values('importance', ascending=False)
                    results['feature_importance'] = feature_importance
                
                # Generate summary plot for first class
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values[0], X, feature_names=feature_names, show=False)
                plt.title('SHAP Summary Plot (Class 0)')
                plt.tight_layout()
                summary_path = os.path.join(self.output_dir, 'shap_summary_class0.png')
                plt.savefig(summary_path)
                plt.close()
                results['summary_plot_path'] = summary_path
            else:
                # For regression or binary classification
                if not isinstance(shap_values, list):
                    shap_values_array = shap_values
                else:
                    shap_values_array = shap_values[0]
                
                # Feature importance based on mean absolute SHAP values
                if feature_names is not None:
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': np.abs(shap_values_array).mean(0)
                    }).sort_values('importance', ascending=False)
                    results['feature_importance'] = feature_importance
                
                # Generate summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values_array, X, feature_names=feature_names, show=False)
                plt.title('SHAP Summary Plot')
                plt.tight_layout()
                summary_path = os.path.join(self.output_dir, 'shap_summary.png')
                plt.savefig(summary_path)
                plt.close()
                results['summary_plot_path'] = summary_path
            
            # Generate bar plot
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap.summary_plot(np.abs(np.array(shap_values)).mean(0), X, 
                                feature_names=feature_names, plot_type='bar', show=False)
            else:
                shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
                shap.summary_plot(shap_values_to_plot, X, 
                                feature_names=feature_names, plot_type='bar', show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            bar_path = os.path.join(self.output_dir, 'shap_importance.png')
            plt.savefig(bar_path)
            plt.close()
            results['importance_plot_path'] = bar_path
            
            # Generate dependence plots for top features
            if feature_names is not None:
                top_features = results.get('feature_importance', pd.DataFrame()).head(5)['feature'].tolist()
                dependence_paths = []
                
                for feature in top_features:
                    plt.figure(figsize=(10, 6))
                    feature_idx = feature_names.index(feature)
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap.dependence_plot(feature_idx, shap_values[0], X, 
                                          feature_names=feature_names, show=False)
                    else:
                        shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
                        shap.dependence_plot(feature_idx, shap_values_to_plot, X, 
                                          feature_names=feature_names, show=False)
                    plt.title(f'SHAP Dependence Plot: {feature}')
                    plt.tight_layout()
                    dep_path = os.path.join(self.output_dir, f'shap_dependence_{feature}.png')
                    plt.savefig(dep_path)
                    plt.close()
                    dependence_paths.append(dep_path)
                
                results['dependence_plot_paths'] = dependence_paths
        
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
        
        return results
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate feature importance using multiple methods.
        
        Args:
            X: Feature data
            y: Target data (optional, required for permutation importance)
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature importance results
        """
        logger.info("Calculating feature importance")
        
        results = {}
        
        # Use column names if feature_names not provided
        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        
        try:
            # Built-in feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                builtin_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                results['builtin_importance'] = builtin_importance
                
                # Plot built-in feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=builtin_importance.head(20))
                plt.title('Built-in Feature Importance')
                plt.tight_layout()
                builtin_path = os.path.join(self.output_dir, 'builtin_importance.png')
                plt.savefig(builtin_path)
                plt.close()
                results['builtin_plot_path'] = builtin_path
            
            # Permutation importance (if y is provided)
            if y is not None and hasattr(self.model, 'predict'):
                try:
                    perm_importance = permutation_importance(
                        self.model, X, y, n_repeats=10, random_state=42, n_jobs=-1
                    )
                    
                    perm_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': perm_importance.importances_mean
                    }).sort_values('importance', ascending=False)
                    results['permutation_importance'] = perm_importance_df
                    
                    # Plot permutation importance
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x='importance', y='feature', data=perm_importance_df.head(20))
                    plt.title('Permutation Feature Importance')
                    plt.tight_layout()
                    perm_path = os.path.join(self.output_dir, 'permutation_importance.png')
                    plt.savefig(perm_path)
                    plt.close()
                    results['permutation_plot_path'] = perm_path
                except Exception as e:
                    logger.error(f"Error calculating permutation importance: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
        
        return results
    
    def analyze_feature_interactions(self, X: pd.DataFrame, 
                                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze feature interactions.
        
        Args:
            X: Feature data
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature interaction results
        """
        logger.info("Analyzing feature interactions")
        
        results = {}
        
        # Use column names if feature_names not provided
        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        
        try:
            # For tree-based models, use SHAP interaction values
            if self.is_tree_model and self.shap_explainer is not None:
                try:
                    # Calculate SHAP interaction values
                    # Note: This can be computationally expensive
                    sample_size = min(500, len(X))
                    X_sample = X.sample(sample_size, random_state=42)
                    
                    interaction_values = self.shap_explainer.shap_interaction_values(X_sample)
                    results['interaction_values'] = interaction_values
                    
                    # Calculate mean absolute interaction values
                    if isinstance(interaction_values, list):
                        # For multi-class models
                        mean_abs_interaction = np.abs(interaction_values[0]).mean(0)
                    else:
                        mean_abs_interaction = np.abs(interaction_values).mean(0)
                    
                    # Create interaction matrix
                    interaction_matrix = pd.DataFrame(
                        mean_abs_interaction,
                        index=feature_names,
                        columns=feature_names
                    )
                    results['interaction_matrix'] = interaction_matrix
                    
                    # Plot interaction matrix
                    plt.figure(figsize=(12, 10))
                    mask = np.zeros_like(interaction_matrix, dtype=bool)
                    mask[np.triu_indices_from(mask)] = True
                    sns.heatmap(interaction_matrix, mask=mask, cmap='viridis', 
                              annot=False, square=True)
                    plt.title('Feature Interaction Strength (SHAP)')
                    plt.tight_layout()
                    interaction_path = os.path.join(self.output_dir, 'feature_interactions.png')
                    plt.savefig(interaction_path)
                    plt.close()
                    results['interaction_plot_path'] = interaction_path
                    
                    # Identify top interactions
                    interaction_values_flat = []
                    for i in range(len(feature_names)):
                        for j in range(i+1, len(feature_names)):
                            interaction_values_flat.append({
                                'feature1': feature_names[i],
                                'feature2': feature_names[j],
                                'interaction_strength': interaction_matrix.iloc[i, j]
                            })
                    
                    top_interactions = pd.DataFrame(interaction_values_flat).sort_values(
                        'interaction_strength', ascending=False
                    ).head(20)
                    results['top_interactions'] = top_interactions
                    
                    # Plot top interactions
                    plt.figure(figsize=(12, 8))
                    sns.barplot(
                        x='interaction_strength', 
                        y=top_interactions.apply(lambda x: f"{x['feature1']} × {x['feature2']}", axis=1),
                        data=top_interactions.head(10)
                    )
                    plt.title('Top 10 Feature Interactions')
                    plt.tight_layout()
                    top_interaction_path = os.path.join(self.output_dir, 'top_interactions.png')
                    plt.savefig(top_interaction_path)
                    plt.close()
                    results['top_interaction_plot_path'] = top_interaction_path
                
                except Exception as e:
                    logger.error(f"Error calculating SHAP interaction values: {str(e)}")
            
            # For non-tree models, use correlation matrix as a simpler alternative
            correlation_matrix = X.corr()
            results['correlation_matrix'] = correlation_matrix
            
            # Plot correlation matrix
            plt.figure(figsize=(12, 10))
            mask = np.zeros_like(correlation_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', 
                      annot=False, square=True, vmin=-1, vmax=1)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            corr_path = os.path.join(self.output_dir, 'correlation_matrix.png')
            plt.savefig(corr_path)
            plt.close()
            results['correlation_plot_path'] = corr_path
        
        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {str(e)}")
        
        return results
    
    def generate_partial_dependence_plots(self, X: pd.DataFrame, 
                                        feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate partial dependence plots for key features.
        
        Args:
            X: Feature data
            feature_names: List of feature names
            
        Returns:
            Dictionary with partial dependence results
        """
        logger.info("Generating partial dependence plots")
        
        results = {}
        
        # Use column names if feature_names not provided
        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        
        try:
            # Check if model is compatible with scikit-learn's partial_dependence
            if hasattr(self.model, 'predict') or hasattr(self.model, 'predict_proba'):
                # Get top features (either from SHAP or built-in importance)
                top_features = []
                
                if hasattr(self.model, 'feature_importances_'):
                    # Use built-in feature importance
                    importance = self.model.feature_importances_
                    top_indices = np.argsort(importance)[::-1][:5]
                    top_features = [feature_names[i] for i in top_indices]
                elif hasattr(self, 'shap_explainer') and self.shap_explainer is not None:
                    # Use SHAP values if available
                    shap_values = self.shap_explainer.shap_values(X.iloc[:100])
                    if isinstance(shap_values, list):
                        mean_abs_shap = np.abs(np.array(shap_values)).mean(0).mean(0)
                    else:
                        mean_abs_shap = np.abs(shap_values).mean(0)
                    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
                    top_features = [feature_names[i] for i in top_indices]
                else:
                    # Use first 5 features
                    top_features = feature_names[:5]
                
                results['top_features'] = top_features
                
                # Generate individual partial dependence plots
                pdp_paths = []
                for feature in top_features:
                    try:
                        # Calculate partial dependence
                        feature_idx = feature_names.index(feature)
                        
                        # Generate plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Use scikit-learn's PartialDependenceDisplay
                        display = PartialDependenceDisplay.from_estimator(
                            self.model, X, [feature_idx], 
                            feature_names=feature_names,
                            ax=ax
                        )
                        
                        plt.title(f'Partial Dependence Plot: {feature}')
                        plt.tight_layout()
                        pdp_path = os.path.join(self.output_dir, f'pdp_{feature}.png')
                        plt.savefig(pdp_path)
                        plt.close()
                        pdp_paths.append(pdp_path)
                    except Exception as e:
                        logger.error(f"Error generating PDP for feature {feature}: {str(e)}")
                
                results['pdp_paths'] = pdp_paths
                
                # Generate 2D partial dependence plots for top feature pairs
                if len(top_features) >= 2:
                    pdp_2d_paths = []
                    for i in range(min(3, len(top_features))):
                        for j in range(i+1, min(4, len(top_features))):
                            try:
                                feature1 = top_features[i]
                                feature2 = top_features[j]
                                feature1_idx = feature_names.index(feature1)
                                feature2_idx = feature_names.index(feature2)
                                
                                # Generate plot
                                fig, ax = plt.subplots(figsize=(8, 6))
                                
                                # Use scikit-learn's PartialDependenceDisplay
                                display = PartialDependenceDisplay.from_estimator(
                                    self.model, X, [(feature1_idx, feature2_idx)], 
                                    feature_names=feature_names,
                                    ax=ax
                                )
                                
                                plt.title(f'2D Partial Dependence: {feature1} vs {feature2}')
                                plt.tight_layout()
                                pdp_2d_path = os.path.join(self.output_dir, f'pdp_2d_{feature1}_{feature2}.png')
                                plt.savefig(pdp_2d_path)
                                plt.close()
                                pdp_2d_paths.append(pdp_2d_path)
                            except Exception as e:
                                logger.error(f"Error generating 2D PDP for features {feature1} and {feature2}: {str(e)}")
                    
                    results['pdp_2d_paths'] = pdp_2d_paths
        
        except Exception as e:
            logger.error(f"Error generating partial dependence plots: {str(e)}")
        
        return results
    
    def explain_prediction(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction.
        
        Args:
            X_instance: Single instance feature data
            
        Returns:
            Dictionary with explanation results
        """
        logger.info("Generating explanation for single prediction")
        
        results = {}
        
        try:
            # Ensure X_instance is 2D
            if len(X_instance.shape) == 1 or (hasattr(X_instance, 'ndim') and X_instance.ndim == 1):
                X_instance = X_instance.values.reshape(1, -1)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(X_instance)
                results['prediction_proba'] = prediction
                prediction_class = self.model.predict(X_instance)
                results['prediction_class'] = prediction_class
            else:
                prediction = self.model.predict(X_instance)
                results['prediction'] = prediction
            
            # Generate SHAP explanation
            if self.shap_explainer is not None:
                # Calculate SHAP values
                shap_values = self.shap_explainer.shap_values(X_instance)
                results['shap_values'] = shap_values
                
                # Generate force plot
                if isinstance(shap_values, list):
                    # For multi-class models
                    predicted_class = prediction_class[0] if hasattr(prediction_class, '__getitem__') else prediction_class
                    shap_values_to_plot = shap_values[predicted_class]
                else:
                    shap_values_to_plot = shap_values
                
                # Create and save force plot
                plt.figure(figsize=(12, 4))
                force_plot = shap.force_plot(
                    self.shap_explainer.expected_value if not isinstance(self.shap_explainer.expected_value, list)
                    else self.shap_explainer.expected_value[0],
                    shap_values_to_plot,
                    X_instance,
                    matplotlib=True,
                    show=False
                )
                plt.title('SHAP Force Plot')
                plt.tight_layout()
                force_path = os.path.join(self.output_dir, 'force_plot.png')
                plt.savefig(force_path)
                plt.close()
                results['force_plot_path'] = force_path
                
                # Create waterfall plot
                plt.figure(figsize=(12, 8))
                shap.plots._waterfall.waterfall_legacy(
                    self.shap_explainer.expected_value if not isinstance(self.shap_explainer.expected_value, list)
                    else self.shap_explainer.expected_value[0],
                    shap_values_to_plot[0],
                    feature_names=X_instance.columns.tolist() if hasattr(X_instance, 'columns') else None,
                    show=False
                )
                plt.title('SHAP Waterfall Plot')
                plt.tight_layout()
                waterfall_path = os.path.join(self.output_dir, 'waterfall_plot.png')
                plt.savefig(waterfall_path)
                plt.close()
                results['waterfall_plot_path'] = waterfall_path
        
        except Exception as e:
            logger.error(f"Error generating prediction explanation: {str(e)}")
        
        return results

# Function to create explainer for any model
def create_model_explainer(model, model_type=None, output_dir=None):
    """
    Create a ModelExplainer instance for the given model.
    
    Args:
        model: Trained model object
        model_type: Type of model ('classifier', 'regressor', 'deep_learning', etc.)
        output_dir: Directory to save explainability artifacts
        
    Returns:
        ModelExplainer instance
    """
    return ModelExplainer(model, model_type, output_dir)
