import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from system_trader.core.explainability.shap_engine import ShapExplainabilityEngine


def create_sample_model_and_data():
    # Generate a simple classification dataset
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    # Train a simple classifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(df, y)
    return model, df


def test_compute_shap_creates_summary_and_json(tmp_path):
    model, df = create_sample_model_and_data()
    output_dir = tmp_path / "shap_outputs"
    engine = ShapExplainabilityEngine({"output_dir": str(output_dir)})

    # Compute SHAP values
    shap_values, summary_plot_path = engine.compute_shap(model, df, model_name="test_model")

    # Assertions on SHAP values object
    assert hasattr(shap_values, "values"), "SHAP values object must have 'values' attribute"
    assert isinstance(shap_values.values, np.ndarray), "SHAP values should be a numpy array"

    # Check that the summary plot file was created
    assert os.path.exists(summary_plot_path), f"Summary plot not found at {summary_plot_path}"

    # Check that the JSON summary file was created and has correct keys
    summary_json_path = output_dir / "test_model_shap_summary.json"
    assert summary_json_path.exists(), "Summary JSON file was not created"

    with open(summary_json_path, 'r') as f:
        summary_dict = json.load(f)

    # The JSON keys should match feature names
    assert set(summary_dict.keys()) == set(df.columns), "JSON summary keys do not match feature names"
    # Values should be numeric and length equal to number of features
    assert all(isinstance(v, float) for v in summary_dict.values()), "JSON summary values should be floats"
    assert len(summary_dict) == df.shape[1], "JSON summary length mismatch with feature count"


def test_plot_interactions_creates_heatmap(tmp_path):
    model, df = create_sample_model_and_data()
    output_dir = tmp_path / "shap_outputs_interactions"
    engine = ShapExplainabilityEngine({"output_dir": str(output_dir)})

    # Generate interaction heatmap
    heatmap_paths = engine.plot_interactions(model, df, model_name="interaction_test")

    assert isinstance(heatmap_paths, list), "plot_interactions should return a list of paths"
    assert len(heatmap_paths) == 1, "Expected exactly one heatmap file"

    heatmap_path = heatmap_paths[0]
    assert os.path.exists(heatmap_path), f"Heatmap file not found at {heatmap_path}"


def test_plot_partial_dependence_creates_plots(tmp_path):
    model, df = create_sample_model_and_data()
    output_dir = tmp_path / "pdp_outputs"
    engine = ShapExplainabilityEngine({"output_dir": str(output_dir)})

    # Select a subset of features to plot PDP
    features = [df.columns[0], df.columns[1]]
    pdp_paths = engine.plot_partial_dependence(model, df, model_name="pdp_test", features=features)

    assert isinstance(pdp_paths, list), "plot_partial_dependence should return a list of paths"
    assert len(pdp_paths) == len(features), "Number of PDP plots should match number of features"

    # Check each PDP file exists and is named correctly
    for feat, path in zip(features, pdp_paths):
        assert os.path.exists(path), f"PDP plot for feature {feat} not found at {path}"
        assert f"pdp_{feat}" in os.path.basename(path), \
            f"PDP plot filename does not include feature name '{feat}'"