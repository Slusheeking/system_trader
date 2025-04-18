"""
MLflow Client Module for Day Trading System

This module provides a mock implementation of MlflowClient to avoid import issues
with the actual MLflow library.
"""

import logging

logger = logging.getLogger(__name__)

class MlflowClient:
    """
    A mock implementation of MlflowClient to avoid import issues.
    This provides the minimum functionality needed for the system to work.
    """
    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri
        logger.info(f"Initialized mock MlflowClient with tracking URI: {tracking_uri}")
        
    def search_model_versions(self, filter_string):
        """Mock implementation that returns an empty list"""
        logger.warning("Using mock MlflowClient.search_model_versions")
        return []
        
    def get_run(self, run_id):
        """Mock implementation that returns None"""
        logger.warning("Using mock MlflowClient.get_run")
        return None
        
    def list_artifacts(self, run_id, path=None):
        """Mock implementation that returns an empty list"""
        logger.warning("Using mock MlflowClient.list_artifacts")
        return []
        
    def get_latest_versions(self, name, stages=None):
        """Mock implementation that returns an empty list"""
        logger.warning("Using mock MlflowClient.get_latest_versions")
        return []
        
    def get_model_version(self, name, version):
        """Mock implementation that returns None"""
        logger.warning("Using mock MlflowClient.get_model_version")
        return None
        
    def create_registered_model(self, name, tags=None, description=None):
        """Mock implementation"""
        logger.warning("Using mock MlflowClient.create_registered_model")
        return None
        
    def set_model_version_tag(self, name, version, key, value):
        """Mock implementation"""
        logger.warning("Using mock MlflowClient.set_model_version_tag")
        return None
        
    def set_registered_model_tag(self, name, key, value):
        """Mock implementation"""
        logger.warning("Using mock MlflowClient.set_registered_model_tag")
        return None
        
    def update_model_version(self, name, version, description=None):
        """Mock implementation"""
        logger.warning("Using mock MlflowClient.update_model_version")
        return None
        
    def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
        """Mock implementation"""
        logger.warning("Using mock MlflowClient.transition_model_version_stage")
        return None
        
    def delete_model_version(self, name, version):
        """Mock implementation"""
        logger.warning("Using mock MlflowClient.delete_model_version")
        return None
        
    def delete_registered_model(self, name):
        """Mock implementation"""
        logger.warning("Using mock MlflowClient.delete_registered_model")
        return None
        
    def get_model_version_download_uri(self, name, version):
        """Mock implementation"""
        logger.warning("Using mock MlflowClient.get_model_version_download_uri")
        return None
        
    def get_model_version_tags(self, name, version):
        """Mock implementation that returns an empty list"""
        logger.warning("Using mock MlflowClient.get_model_version_tags")
        return []
        
    def get_registered_model_tags(self, name):
        """Mock implementation that returns an empty list"""
        logger.warning("Using mock MlflowClient.get_registered_model_tags")
        return []
        
    def search_registered_models(self):
        """Mock implementation that returns an empty list"""
        logger.warning("Using mock MlflowClient.search_registered_models")
        return []
        
    def get_registered_model(self, name):
        """Mock implementation that returns None"""
        logger.warning("Using mock MlflowClient.get_registered_model")
        return None