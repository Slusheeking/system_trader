"""
MLflow Model Registry Module for Day Trading System

This module provides integration with the MLflow Model Registry for managing
model versions, transitions through lifecycle stages, and model deployment.
Focuses on reproducibility, versioning, and governance for trading models.
"""

import os
import yaml
import mlflow
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import datetime
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    A wrapper around MLflow Model Registry for managing models
    in the day trading system.
    """
    
    def __init__(self, tracking_uri: str = None):
        """
        Initialize the MLflow Model Registry wrapper.
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Using MLflow tracking URI: {tracking_uri}")
        else:
            logger.info(f"Using default MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Initialize MLflow client
        self.client = MlflowClient()
        
        # Define model stages
        self.STAGES = {
            'dev': 'Development',
            'staging': 'Staging',
            'prod': 'Production',
            'archived': 'Archived'
        }
    
    def register_model(self, run_id: str, model_path: str, name: str) -> str:
        """
        Register a model from a run artifact.
        
        Args:
            run_id: ID of the run containing the model artifact
            model_path: Path to the model within the run's artifacts
            name: Name to register the model under
            
        Returns:
            Version of the registered model
        """
        try:
            result = mlflow.register_model(
                f"runs:/{run_id}/{model_path}",
                name
            )
            model_version = result.version
            
            logger.info(f"Registered model {name} version {model_version} from run {run_id}")
            return model_version
            
        except MlflowException as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def create_registered_model(self, name: str, tags: Dict[str, str] = None, 
                               description: str = None) -> None:
        """
        Create a new registered model.
        
        Args:
            name: Name for the registered model
            tags: Dictionary of tags
            description: Description for the model
        """
        try:
            self.client.create_registered_model(
                name=name,
                tags=tags,
                description=description
            )
            logger.info(f"Created new registered model: {name}")
            
        except MlflowException as e:
            # If the model already exists, log and continue
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                logger.info(f"Registered model {name} already exists")
            else:
                logger.error(f"Error creating registered model: {str(e)}")
                raise
    
    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        """
        Set a tag on a specific model version.
        
        Args:
            name: Name of the registered model
            version: Version number
            key: Tag key
            value: Tag value
        """
        try:
            self.client.set_model_version_tag(
                name=name,
                version=version,
                key=key,
                value=value
            )
            logger.debug(f"Set tag {key}={value} on model {name} version {version}")
            
        except MlflowException as e:
            logger.error(f"Error setting model version tag: {str(e)}")
            raise
    
    def set_model_tag(self, name: str, key: str, value: str) -> None:
        """
        Set a tag on a registered model.
        
        Args:
            name: Name of the registered model
            key: Tag key
            value: Tag value
        """
        try:
            self.client.set_registered_model_tag(
                name=name,
                key=key,
                value=value
            )
            logger.debug(f"Set tag {key}={value} on model {name}")
            
        except MlflowException as e:
            logger.error(f"Error setting model tag: {str(e)}")
            raise
    
    def update_model_version(self, name: str, version: str, description: str = None) -> None:
        """
        Update a model version's description.
        
        Args:
            name: Name of the registered model
            version: Version number
            description: New description
        """
        try:
            self.client.update_model_version(
                name=name,
                version=version,
                description=description
            )
            logger.info(f"Updated description for model {name} version {version}")
            
        except MlflowException as e:
            logger.error(f"Error updating model version: {str(e)}")
            raise
    
    def transition_model_version_stage(self, name: str, version: str, 
                                     stage: str, archive_existing_versions: bool = False) -> None:
        """
        Transition a model version to a different stage.
        
        Args:
            name: Name of the registered model
            version: Version number
            stage: Target stage (one of 'Development', 'Staging', 'Production', 'Archived')
            archive_existing_versions: Whether to archive versions in the target stage
        """
        # Validate stage
        if stage not in self.STAGES.values():
            valid_stages = list(self.STAGES.values())
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
        
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            
            # If archiving existing versions, log what was affected
            if archive_existing_versions:
                logger.info(f"Transitioned model {name} version {version} to {stage} "
                           f"and archived existing versions in {stage}")
            else:
                logger.info(f"Transitioned model {name} version {version} to {stage}")
                
        except MlflowException as e:
            logger.error(f"Error transitioning model version stage: {str(e)}")
            raise
    
    def delete_model_version(self, name: str, version: str) -> None:
        """
        Delete a model version.
        
        Args:
            name: Name of the registered model
            version: Version number
        """
        try:
            self.client.delete_model_version(
                name=name,
                version=version
            )
            logger.info(f"Deleted model {name} version {version}")
            
        except MlflowException as e:
            logger.error(f"Error deleting model version: {str(e)}")
            raise
    
    def delete_registered_model(self, name: str) -> None:
        """
        Delete a registered model and all its versions.
        
        Args:
            name: Name of the registered model
        """
        try:
            self.client.delete_registered_model(name=name)
            logger.info(f"Deleted registered model {name} and all its versions")
            
        except MlflowException as e:
            logger.error(f"Error deleting registered model: {str(e)}")
            raise
    
    def get_latest_model_version(self, name: str, stage: str = None) -> Optional[str]:
        """
        Get the latest version of a model, optionally in a specific stage.
        
        Args:
            name: Name of the registered model
            stage: Optional stage to filter by
            
        Returns:
            Latest version number or None if not found
        """
        try:
            # Validate stage if provided
            if stage and stage not in self.STAGES.values():
                valid_stages = list(self.STAGES.values())
                raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
            
            # Get all versions of the model
            versions = self.client.get_latest_versions(name, stages=[stage] if stage else None)
            
            if not versions:
                logger.warning(f"No versions found for model {name}" + 
                              (f" in stage {stage}" if stage else ""))
                return None
            
            # Sort versions by version number (descending) and return the first one
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            return latest_version.version
            
        except MlflowException as e:
            logger.error(f"Error getting latest model version: {str(e)}")
            return None
    
    def get_model_download_uri(self, name: str, version: str = None, stage: str = None) -> str:
        """
        Get the download URI for a model version.
        
        Args:
            name: Name of the registered model
            version: Version number, or None to use latest
            stage: Stage to get the model from, or None to get specified version
            
        Returns:
            URI for downloading the model
        """
        # Check that exactly one of version or stage is provided
        if (version is None and stage is None) or (version is not None and stage is not None):
            raise ValueError("Exactly one of version or stage must be provided")
            
        try:
            if stage:
                # Validate stage
                if stage not in self.STAGES.values():
                    valid_stages = list(self.STAGES.values())
                    raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
                
                # Get latest version in stage
                versions = self.client.get_latest_versions(name, stages=[stage])
                if not versions:
                    raise ValueError(f"No versions found for model {name} in stage {stage}")
                
                version_info = versions[0]
                version = version_info.version
            
            # Get the model version
            model_version = self.client.get_model_version(name, version)
            
            # Return the download URI
            return model_version.source
            
        except MlflowException as e:
            logger.error(f"Error getting model download URI: {str(e)}")
            raise
    
    def load_model(self, name: str, version: str = None, stage: str = None) -> Any:
        """
        Load a model from the registry.
        
        Args:
            name: Name of the registered model
            version: Version number, or None to use latest
            stage: Stage to load the model from, or None to load specified version
            
        Returns:
            Loaded model
        """
        # Check that exactly one of version or stage is provided
        if (version is None and stage is None) or (version is not None and stage is not None):
            raise ValueError("Exactly one of version or stage must be provided")
            
        try:
            # Construct the model URI
            if stage:
                # Validate stage
                if stage not in self.STAGES.values():
                    valid_stages = list(self.STAGES.values())
                    raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
                
                model_uri = f"models:/{name}/{stage}"
                logger.info(f"Loading model {name} from stage {stage}")
            else:
                model_uri = f"models:/{name}/{version}"
                logger.info(f"Loading model {name} version {version}")
            
            # Load the model
            # First try as pyfunc, then fall back to specific flavors
            try:
                model = mlflow.pyfunc.load_model(model_uri)
            except MlflowException:
                # Try to determine model flavor and load accordingly
                try:
                    # Get the model artifacts and check flavor
                    model_info = self.client.get_model_version_download_uri(name, version)
                    model_path = model_info.replace('runs:/', '')
                    
                    # Try each flavor
                    for flavor in [mlflow.xgboost, mlflow.sklearn, mlflow.tensorflow, mlflow.pyfunc]:
                        try:
                            model = flavor.load_model(model_uri)
                            return model
                        except Exception:
                            continue
                            
                    raise ValueError(f"Could not determine model flavor for {name} {version}")
                        
                except Exception as e:
                    logger.error(f"Error loading model using specific flavor: {str(e)}")
                    raise
            
            return model
            
        except MlflowException as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self, name: str, version: str = None, stage: str = None) -> Dict[str, Any]:
        """
        Get detailed information about a model version.
        
        Args:
            name: Name of the registered model
            version: Version number, or None to use latest
            stage: Stage to get info from, or None to get specified version
            
        Returns:
            Dictionary with model information
        """
        # Check that exactly one of version or stage is provided
        if (version is None and stage is None) or (version is not None and stage is not None):
            raise ValueError("Exactly one of version or stage must be provided")
            
        try:
            if stage:
                # Validate stage
                if stage not in self.STAGES.values():
                    valid_stages = list(self.STAGES.values())
                    raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
                
                # Get latest version in stage
                versions = self.client.get_latest_versions(name, stages=[stage])
                if not versions:
                    raise ValueError(f"No versions found for model {name} in stage {stage}")
                
                version_info = versions[0]
                version = version_info.version
            
            # Get the model version
            model_version = self.client.get_model_version(name, version)
            
            # Get model version details
            info = {
                "name": model_version.name,
                "version": model_version.version,
                "creation_timestamp": datetime.datetime.fromtimestamp(model_version.creation_timestamp/1000.0),
                "last_updated_timestamp": datetime.datetime.fromtimestamp(model_version.last_updated_timestamp/1000.0),
                "user_id": model_version.user_id,
                "current_stage": model_version.current_stage,
                "description": model_version.description,
                "source": model_version.source,
                "run_id": model_version.run_id,
                "status": model_version.status,
                "status_message": model_version.status_message,
                "tags": {tag.key: tag.value for tag in self.client.get_model_version_tags(name, version)}
            }
            
            # Get the run information if available
            if model_version.run_id:
                try:
                    run = mlflow.get_run(model_version.run_id)
                    
                    # Add run information
                    info["run"] = {
                        "status": run.info.status,
                        "start_time": datetime.datetime.fromtimestamp(run.info.start_time/1000.0),
                        "end_time": datetime.datetime.fromtimestamp(run.info.end_time/1000.0) if run.info.end_time else None,
                        "artifact_uri": run.info.artifact_uri,
                        "metrics": run.data.metrics,
                        "params": run.data.params,
                        "tags": run.data.tags
                    }
                except Exception as e:
                    logger.warning(f"Could not get run information: {str(e)}")
                    info["run"] = None
            
            return info
            
        except MlflowException as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise
    
    def get_all_registered_models(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered models.
        
        Returns:
            List of dictionaries with model information
        """
        try:
            # Get all registered models
            models = self.client.search_registered_models()
            
            # Format the results
            result = []
            for model in models:
                model_info = {
                    "name": model.name,
                    "creation_timestamp": datetime.datetime.fromtimestamp(model.creation_timestamp/1000.0),
                    "last_updated_timestamp": datetime.datetime.fromtimestamp(model.last_updated_timestamp/1000.0),
                    "user_id": model.user_id,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "creation_timestamp": datetime.datetime.fromtimestamp(v.creation_timestamp/1000.0),
                            "last_updated_timestamp": datetime.datetime.fromtimestamp(v.last_updated_timestamp/1000.0),
                            "run_id": v.run_id
                        }
                        for v in model.latest_versions
                    ],
                    "tags": {tag.key: tag.value for tag in self.client.get_registered_model_tags(model.name)}
                }
                result.append(model_info)
            
            return result
            
        except MlflowException as e:
            logger.error(f"Error getting registered models: {str(e)}")
            return []
    
    def compare_model_versions(self, name: str, versions: List[str]) -> pd.DataFrame:
        """
        Compare metrics for different versions of a model.
        
        Args:
            name: Name of the registered model
            versions: List of version numbers to compare
            
        Returns:
            DataFrame with metric comparisons
        """
        try:
            # Initialize results
            results = []
            
            # Get info for each version
            for version in versions:
                try:
                    info = self.get_model_info(name, version=version)
                    
                    # Skip if run information is not available
                    if not info.get("run"):
                        logger.warning(f"No run information available for model {name} version {version}")
                        continue
                    
                    # Extract model stage and metrics
                    row = {
                        "version": version,
                        "stage": info["current_stage"],
                        "creation_time": info["creation_timestamp"],
                        "run_id": info["run_id"],
                    }
                    
                    # Add metrics
                    metrics = info["run"]["metrics"]
                    for key, value in metrics.items():
                        row[f"metric_{key}"] = value
                    
                    results.append(row)
                    
                except Exception as e:
                    logger.warning(f"Error getting info for model {name} version {version}: {str(e)}")
                    continue
            
            # Convert to DataFrame
            if not results:
                logger.warning(f"No comparison data available for model {name}")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            
            # Sort by version
            df = df.sort_values("version", ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error comparing model versions: {str(e)}")
            return pd.DataFrame()
    
    def export_model_details(self, name: str, output_dir: str) -> str:
        """
        Export detailed information about a model and all its versions.
        
        Args:
            name: Name of the registered model
            output_dir: Directory to save the export
            
        Returns:
            Path to the exported file
        """
        try:
            # Get registered model
            model = self.client.get_registered_model(name)
            
            # Get all versions
            versions = self.client.search_model_versions(f"name='{name}'")
            
            # Create export data structure
            export_data = {
                "name": model.name,
                "creation_timestamp": datetime.datetime.fromtimestamp(model.creation_timestamp/1000.0).isoformat(),
                "last_updated_timestamp": datetime.datetime.fromtimestamp(model.last_updated_timestamp/1000.0).isoformat(),
                "user_id": model.user_id,
                "description": model.description,
                "tags": {tag.key: tag.value for tag in self.client.get_registered_model_tags(name)},
                "versions": []
            }
            
            # Add version details
            for version in versions:
                version_data = {
                    "version": version.version,
                    "current_stage": version.current_stage,
                    "creation_timestamp": datetime.datetime.fromtimestamp(version.creation_timestamp/1000.0).isoformat(),
                    "last_updated_timestamp": datetime.datetime.fromtimestamp(version.last_updated_timestamp/1000.0).isoformat(),
                    "user_id": version.user_id,
                    "description": version.description,
                    "source": version.source,
                    "run_id": version.run_id,
                    "status": version.status,
                    "status_message": version.status_message,
                    "tags": {tag.key: tag.value for tag in self.client.get_model_version_tags(name, version.version)}
                }
                
                # Get run information if available
                if version.run_id:
                    try:
                        run = mlflow.get_run(version.run_id)
                        
                        # Add run information
                        version_data["run"] = {
                            "status": run.info.status,
                            "start_time": datetime.datetime.fromtimestamp(run.info.start_time/1000.0).isoformat() if run.info.start_time else None,
                            "end_time": datetime.datetime.fromtimestamp(run.info.end_time/1000.0).isoformat() if run.info.end_time else None,
                            "artifact_uri": run.info.artifact_uri,
                            "metrics": run.data.metrics,
                            "params": run.data.params,
                            "tags": run.data.tags
                        }
                    except Exception as e:
                        logger.warning(f"Could not get run information for version {version.version}: {str(e)}")
                        version_data["run"] = None
                
                export_data["versions"].append(version_data)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Create export filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_export_{timestamp}.yaml"
            filepath = os.path.join(output_dir, filename)
            
            # Write the export data to file
            with open(filepath, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Exported model details for {name} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting model details: {str(e)}")
            raise
    
    def import_model(self, run_id: str, model_path: str, name: str, 
                    version_description: str = None, tags: Dict[str, str] = None,
                    stage: str = None) -> str:
        """
        Import a model from a run to the registry with optional staging.
        
        Args:
            run_id: ID of the run containing the model
            model_path: Path to the model within the run's artifacts
            name: Name to register the model under
            version_description: Description for the model version
            tags: Tags to add to the model version
            stage: Stage to transition the model to after registration
            
        Returns:
            Version of the registered model
        """
        try:
            # Register the model
            version = self.register_model(run_id, model_path, name)
            
            # Add description if provided
            if version_description:
                self.update_model_version(name, version, description=version_description)
            
            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.set_model_version_tag(name, version, key, value)
            
            # Transition to stage if provided
            if stage:
                self.transition_model_version_stage(name, version, stage)
            
            return version
            
        except Exception as e:
            logger.error(f"Error importing model: {str(e)}")
            raise

# Helper functions for common registry operations

def register_trading_model(run_id: str, model_path: str, model_type: str, 
                         name: Optional[str] = None, 
                         version_description: Optional[str] = None,
                         tags: Optional[Dict[str, str]] = None,
                         stage: Optional[str] = None,
                         registry_uri: Optional[str] = None) -> Tuple[str, str]:
    """
    Register a trading model with appropriate metadata.
    
    Args:
        run_id: MLflow run ID containing the model
        model_path: Path to the model within the run's artifacts
        model_type: Type of trading model (e.g., 'stock_selection', 'entry_timing')
        name: Optional name override, otherwise constructed from model_type
        version_description: Description for the model version
        tags: Additional tags for the model
        stage: Stage to transition to after registration
        registry_uri: URI for the MLflow registry
        
    Returns:
        Tuple of (model_name, version)
    """
    # Initialize registry client
    registry = ModelRegistry(tracking_uri=registry_uri)
    
    # Construct model name if not provided
    if name is None:
        name = f"trading_{model_type}"
    
    # Prepare base tags
    base_tags = {
        "model_type": model_type,
        "registered_by": os.environ.get("USER", "unknown"),
        "registered_at": datetime.datetime.now().isoformat(),
    }
    
    # Merge with provided tags
    if tags:
        base_tags.update(tags)
    
    # Create registered model if it doesn't exist
    try:
        registry.create_registered_model(
            name=name,
            tags={"model_type": model_type},
            description=f"Trading model for {model_type}"
        )
    except:
        # Model already exists, continue
        pass
    
    # Register the model
    version = registry.import_model(
        run_id=run_id,
        model_path=model_path,
        name=name,
        version_description=version_description,
        tags=base_tags,
        stage=stage
    )
    
    logger.info(f"Registered trading model {name} version {version}")
    return name, version

def get_production_model(model_type: str, registry_uri: Optional[str] = None):
    """
    Get the current production model of a specific type.
    
    Args:
        model_type: Type of trading model (e.g., 'stock_selection', 'entry_timing')
        registry_uri: URI for the MLflow registry
        
    Returns:
        Loaded model or None if not found
    """
    # Initialize registry client
    registry = ModelRegistry(tracking_uri=registry_uri)
    
    # Construct model name
    name = f"trading_{model_type}"
    
    try:
        # Load the model from Production stage
        model = registry.load_model(name=name, stage="Production")
        logger.info(f"Loaded production model for {model_type}")
        return model
    except Exception as e:
        logger.warning(f"No production model available for {model_type}: {str(e)}")
        return None

def promote_model_to_production(model_name: str, version: str, 
                               archive_existing: bool = True,
                               registry_uri: Optional[str] = None) -> bool:
    """
    Promote a model version to production.
    
    Args:
        model_name: Name of the registered model
        version: Version to promote
        archive_existing: Whether to archive existing production versions
        registry_uri: URI for the MLflow registry
        
    Returns:
        True if successful, False otherwise
    """
    # Initialize registry client
    registry = ModelRegistry(tracking_uri=registry_uri)
    
    try:
        # Add transition tag with timestamp
        registry.set_model_version_tag(
            name=model_name,
            version=version,
            key="promoted_to_production_at",
            value=datetime.datetime.now().isoformat()
        )
        
        # Transition to Production stage
        registry.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=archive_existing
        )
        
        logger.info(f"Promoted {model_name} version {version} to Production")
        return True
        
    except Exception as e:
        logger.error(f"Failed to promote model to production: {str(e)}")
        return False

def compare_model_performance(model_name: str, versions: List[str] = None,
                             registry_uri: Optional[str] = None) -> pd.DataFrame:
    """
    Compare performance metrics of different model versions.
    
    Args:
        model_name: Name of the registered model
        versions: List of versions to compare, or None to get all
        registry_uri: URI for the MLflow registry
        
    Returns:
        DataFrame with comparison metrics
    """
    # Initialize registry client
    registry = ModelRegistry(tracking_uri=registry_uri)
    
    try:
        # If versions not specified, get all versions
        if versions is None:
            all_versions = registry.client.search_model_versions(f"name='{model_name}'")
            versions = [v.version for v in all_versions]
        
        # Get comparison data
        comparison_df = registry.compare_model_versions(model_name, versions)
        
        return comparison_df
        
    except Exception as e:
        logger.error(f"Error comparing model performance: {str(e)}")
        return pd.DataFrame()