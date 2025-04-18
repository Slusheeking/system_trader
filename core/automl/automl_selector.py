# -*- coding: utf-8 -*-
"""
AutoML Selector

Performs automated model architecture search using classical AutoML (auto-sklearn)
or neural tuning (Keras-Tuner) based on the model type.
Configuration is passed via the 'automl' section of the training config.
"""
import logging
from typing import Dict, Any

import pandas as pd

# Try imports; handle missing dependencies gracefully
try:
    import autosklearn.classification as asc
except ImportError:
    asc = None
    # Log a more informative message about auto-sklearn
    import logging
    logging.getLogger(__name__).warning(
        "auto-sklearn not installed. Classical AutoML will fall back to scikit-learn. "
        "To install auto-sklearn, follow instructions at https://automl.github.io/auto-sklearn/master/"
    )

try:
    import keras_tuner as kt
except ImportError:
    try:
        import kerastuner as kt
    except ImportError:
        kt = None
        # Log a more informative message about keras-tuner
        import logging
        logging.getLogger(__name__).warning(
            "keras-tuner not installed. Neural AutoML will be disabled. "
            "To install keras-tuner, run: pip install keras-tuner"
        )

# TensorFlow import for neural model building
tf = None
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.config.experimental.set_visible_devices([], 'GPU')  # avoid GPU issues here
    tf = tf
except ImportError:
    pass

class AutoMLSelector:
    """
    Automated model architecture search for classical and neural models.
    config format:
      classical:
        library: "auto-sklearn"
        time_limits: <seconds>
      neural:
        library: "keras-tuner"
        max_trials: <int>
        executions_per_trial: <int>
        epochs: <int> (optional)
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.classical_cfg = self.config.get('classical', {})
        self.neural_cfg = self.config.get('neural', {})
        self.logger = logging.getLogger(__name__)

        # Define model types considered 'neural'
        self._neural_models = {'entry_timing', 'peak_detection'}

    def search_architecture(
        self,
        data: pd.DataFrame,
        target: str,
        model_type: str = None
    ) -> Dict[str, Any]:
        """
        Perform architecture search and return best parameters.

        Args:
            data: DataFrame containing features and target column
            target: Name of the target column in data
            model_type: Type of model (e.g. 'entry_timing', 'stock_selection')

        Returns:
            Dict of parameters to update trainer config
        """
        if target not in data.columns:
            self.logger.error(f"Target column '{target}' not found.")
            return {}

        # Split features and labels
        X = data.drop(columns=[target])
        y = data[target]

        # Choose search method
        if model_type in self._neural_models:
            return self._search_neural(X, y, model_type)
        else:
            return self._search_classical(X, y)

    def _search_classical(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform AutoML search using auto-sklearn for classical ML models.
        Falls back to scikit-learn's GridSearchCV if auto-sklearn is not available.
        """
        lib = self.classical_cfg.get('library', 'auto-sklearn')
        time_limits = self.classical_cfg.get('time_limits', 3600)
        per_run = self.classical_cfg.get('per_run_time_limit', max(60, time_limits // 10))

        # Try auto-sklearn if available
        if asc is not None and lib == 'auto-sklearn':
            try:
                self.logger.info("Using auto-sklearn for classical model search")
                automl = asc.AutoSklearnClassifier(
                    time_left_for_this_task=int(time_limits),
                    per_run_time_limit=int(per_run)
                )
                automl.fit(X, y)
                # Extract best configuration
                params = automl.get_params()
                # Filter to only relevant hyperparameters (prefix-based)
                best_params = {k: v for k, v in params.items()}
                return best_params
            except Exception as e:
                self.logger.error(f"Error in auto-sklearn search: {e}")
                self.logger.info("Falling back to scikit-learn GridSearchCV")
        else:
            if asc is None:
                self.logger.warning("auto-sklearn not installed, using scikit-learn GridSearchCV instead.")
            elif lib != 'auto-sklearn':
                self.logger.warning(f"Configured classical library '{lib}' not supported, using scikit-learn GridSearchCV instead.")

        # Fallback to scikit-learn's GridSearchCV
        try:
            from sklearn.model_selection import GridSearchCV
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            
            # Define a set of models and parameters to search
            models = {
                'random_forest': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                },
                'svm': {
                    'model': SVC(probability=True, random_state=42),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'logistic_regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'params': {
                        'C': [0.1, 1, 10, 100],
                        'solver': ['liblinear', 'saga']
                    }
                }
            }
            
            # Find the best model and parameters
            best_score = -1
            best_model_name = None
            best_params = {}
            
            for model_name, model_info in models.items():
                self.logger.info(f"Trying {model_name} with GridSearchCV")
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X, y)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model_name = model_name
                    best_params = {
                        'model_type': best_model_name,
                        'best_score': best_score,
                        **grid_search.best_params_
                    }
            
            self.logger.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error in scikit-learn GridSearchCV: {e}")
            return {}

    def _search_neural(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Perform neural architecture search using Keras-Tuner.
        """
        if kt is None or tf is None:
            self.logger.error("Keras-Tuner or TensorFlow not installed, skipping neural AutoML search.")
            return {}

        max_trials = int(self.neural_cfg.get('max_trials', 10))
        execs = int(self.neural_cfg.get('executions_per_trial', 1))
        epochs = int(self.neural_cfg.get('epochs', 10))

        # Create tuner
        tuner = kt.RandomSearch(
            hypermodel=self._get_nn_builder(input_shape=X.shape[1], model_type=model_type),
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=execs,
            directory='automl_nn_tuning',
            project_name=model_type
        )

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y.values,
            test_size=0.2,
            random_state=42
        )

        try:
            tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                verbose=0
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            return best_hps.values

        except Exception as e:
            self.logger.error(f"Error in neural AutoML search: {e}")
            return {}

    def _get_nn_builder(self, input_shape: int, model_type: str):
        """
        Returns a Keras-Tuner builder function based on model_type.
        """
        def build_model(hp):
            model = tf.keras.Sequential()
            # Tune number of layers
            num_layers = hp.Int('num_layers', 1, 3)
            for i in range(num_layers):
                units = hp.Int(f'units_{i}', 16, 128, step=16)
                model.add(tf.keras.layers.Dense(units, activation='relu', input_shape=(input_shape,) if i == 0 else ()))
                dropout = hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)
                model.add(tf.keras.layers.Dropout(dropout))

            # Output layer
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model

        return build_model
