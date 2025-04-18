import os
import json
import numpy as np
import shap
import matplotlib.pyplot as plt
import mlflow
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns


class ShapExplainabilityEngine:
    """
    Engine to compute SHAP values, interaction heatmaps, and partial dependence plots,
    and log the results to MLflow.
    """

    def __init__(self, config: dict):
        """
        Initialize the SHAP explainability engine.

        Args:
            config: Configuration dict, may contain 'output_dir' key for saving artifacts.
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'shap_outputs')
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_shap(self, model, data, model_name: str):
        """
        Compute SHAP values for the given model and data, save summary plot and JSON,
        and log both to MLflow.

        Args:
            model: Trained model with predict or predict_proba method.
            data: DataFrame or numpy array of input features.
            model_name: Name identifier for the model.

        Returns:
            shap_values: SHAP values object.
            summary_plot_path: Path to the saved SHAP summary plot image.
        """
        # Initialize SHAP explainer
        try:
            # Choose appropriate explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, data)
            else:
                explainer = shap.Explainer(model, data)

            # Compute SHAP values
            shap_values = explainer(data)

            # Save SHAP summary plot
            summary_plot_path = os.path.join(
                self.output_dir, f"{model_name}_shap_summary.png"
            )
            plt.figure()
            shap.summary_plot(shap_values, data, show=False)
            plt.savefig(summary_plot_path, bbox_inches='tight')
            plt.close()

            # Create summary JSON of mean absolute SHAP values
            feature_names = (
                data.columns.tolist() if hasattr(data, 'columns') else [f'f{i}' for i in range(shap_values.values.shape[1])]
            )
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            summary_dict = dict(zip(feature_names, mean_abs.tolist()))
            summary_json_path = os.path.join(
                self.output_dir, f"{model_name}_shap_summary.json"
            )
            with open(summary_json_path, 'w') as jf:
                json.dump(summary_dict, jf)

            # Log artifacts to MLflow
            mlflow.log_artifact(summary_plot_path)
            mlflow.log_artifact(summary_json_path)

            return shap_values, summary_plot_path

        except Exception as e:
            raise RuntimeError(f"Error computing SHAP values for {model_name}: {str(e)}")

    def plot_interactions(self, model, data, model_name: str):
        """
        Compute SHAP interaction values, generate a heatmap of mean absolute interactions,
        save the plot, and log it to MLflow.

        Args:
            model: Trained model.
            data: DataFrame or array of input features.
            model_name: Name identifier for the model.

        Returns:
            List containing the path to the interaction heatmap image.
        """
        try:
            # Use TreeExplainer for interaction values if available
            if hasattr(model, 'feature_importances_') or hasattr(model, 'booster'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, data)

            # Compute SHAP interaction values (samples x features x features)
            interaction_values = explainer.shap_interaction_values(data)

            # Aggregate by mean absolute interaction
            # If tuple returned (for classification), take the first
            if isinstance(interaction_values, list):
                # Multiclass: list of arrays per class
                interaction_matrix = np.abs(interaction_values[0]).mean(axis=0)
            else:
                interaction_matrix = np.abs(interaction_values).mean(axis=0)

            # Plot heatmap
            heatmap_path = os.path.join(
                self.output_dir, f"{model_name}_interaction_heatmap.png"
            )
            plt.figure(figsize=self.config.get('heatmap_size', (12, 10)))
            sns.heatmap(
                interaction_matrix,
                xticklabels=(data.columns if hasattr(data, 'columns') else None),
                yticklabels=(data.columns if hasattr(data, 'columns') else None),
                cmap='viridis'
            )
            plt.title(f"SHAP Interaction Heatmap: {model_name}")
            plt.tight_layout()
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()

            # Log to MLflow
            mlflow.log_artifact(heatmap_path)

            return [heatmap_path]

        except Exception as e:
            raise RuntimeError(f"Error generating SHAP interactions for {model_name}: {str(e)}")

    def plot_partial_dependence(self, model, data, model_name: str, features):
        """
        Generate partial dependence plots for specified features using sklearn,
        save each plot, and log them to MLflow.

        Args:
            model: Trained estimator.
            data: DataFrame of input features.
            model_name: Name identifier for the model.
            features: List of feature names or indices.

        Returns:
            List of paths to the saved PDP plot images.
        """
        plot_paths = []
        try:
            for feat in features:
                display = PartialDependenceDisplay.from_estimator(
                    model,
                    data,
                    [feat],
                    kind='average',
                    subsample=self.config.get('pdp_subsample', 1000),
                    random_state=self.config.get('random_state', 42)
                )
                fig = display.figure_
                pdp_path = os.path.join(
                    self.output_dir, f"{model_name}_pdp_{feat}.png"
                )
                fig.savefig(pdp_path, bbox_inches='tight')
                plt.close(fig)

                # Log to MLflow
                mlflow.log_artifact(pdp_path)
                plot_paths.append(pdp_path)

            return plot_paths

        except Exception as e:
            raise RuntimeError(f"Error generating PDP for {model_name}: {str(e)}")
