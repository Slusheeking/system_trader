import logging
import pandas as pd
import numpy as np
import json
import os
import datetime
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta

# For data storage
import pickle
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Enhanced Feature Store for managing features with versioning, lineage tracking,
    and sharing across different model types.
    
    This class provides:
    1. Feature registration and versioning
    2. Feature lineage tracking
    3. Feature metadata management
    4. Feature data storage and retrieval
    5. Feature sharing across model types
    6. Feature discovery and search
    """
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        """
        Initialize the FeatureStore.

        Args:
            config: Optional configuration dictionary or connection string.
        """
        # Handle the case where config is a string (connection string)
        if isinstance(config, str):
            self.config = {'connection_string': config}
        else:
            self.config = config if config is not None else {}
        
        # Storage configuration
        self.storage_dir = self.config.get('storage_dir', 'feature_store_data')
        self.metadata_db_path = os.path.join(self.storage_dir, 'feature_metadata.db')
        self.feature_data_dir = os.path.join(self.storage_dir, 'feature_data')
        
        # Ensure directories exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.feature_data_dir, exist_ok=True)
        
        # Initialize metadata database
        self._initialize_metadata_db()
        
        # Cache for feature metadata
        self.feature_cache = {}
        
        logger.info("Enhanced FeatureStore initialized.")
    
    def _initialize_metadata_db(self):
        """Initialize the SQLite database for feature metadata."""
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # Create features table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                data_type TEXT,
                statistics TEXT,
                tags TEXT,
                owner TEXT,
                status TEXT,
                UNIQUE(name, version)
            )
            ''')
            
            # Create lineage table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id INTEGER NOT NULL,
                source_feature_id INTEGER,
                transformation TEXT,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (feature_id) REFERENCES features(id),
                FOREIGN KEY (source_feature_id) REFERENCES features(id)
            )
            ''')
            
            # Create model_usage table to track which models use which features
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT,
                importance REAL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (feature_id) REFERENCES features(id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Feature metadata database initialized.")
        except Exception as e:
            logger.error(f"Error initializing metadata database: {str(e)}")
            raise
    
    def register_feature(self, 
                        feature_name: str, 
                        feature_metadata: Dict[str, Any], 
                        version: str = "latest",
                        description: str = "",
                        data_type: str = "float",
                        tags: List[str] = None,
                        owner: str = "",
                        status: str = "active") -> int:
        """
        Register a new feature or a new version of an existing feature.

        Args:
            feature_name: The name of the feature.
            feature_metadata: Metadata about the feature (e.g., source, transformation).
            version: The version of the feature.
            description: Description of the feature.
            data_type: Data type of the feature.
            tags: List of tags for categorizing the feature.
            owner: Owner of the feature.
            status: Status of the feature (active, deprecated, etc.).
            
        Returns:
            Feature ID in the database.
        """
        try:
            # Convert tags to JSON string
            tags_json = json.dumps(tags) if tags else json.dumps([])
            
            # Convert statistics to JSON string if present
            if 'statistics' in feature_metadata:
                feature_metadata['statistics'] = json.dumps(feature_metadata['statistics'])
            
            # Convert feature_metadata to JSON string
            metadata_json = json.dumps(feature_metadata)
            
            # Get current timestamp
            now = datetime.now().isoformat()
            
            # Insert into database
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # Check if feature version already exists
            cursor.execute(
                "SELECT id FROM features WHERE name = ? AND version = ?", 
                (feature_name, version)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing feature
                feature_id = existing[0]
                cursor.execute(
                    """
                    UPDATE features 
                    SET description = ?, data_type = ?, statistics = ?, 
                        tags = ?, owner = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (description, data_type, metadata_json, tags_json, 
                     owner, status, now, feature_id)
                )
                logger.info(f"Updated feature '{feature_name}' version '{version}'.")
            else:
                # Insert new feature
                cursor.execute(
                    """
                    INSERT INTO features 
                    (name, version, description, data_type, statistics, 
                     tags, owner, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (feature_name, version, description, data_type, metadata_json,
                     tags_json, owner, status, now, now)
                )
                feature_id = cursor.lastrowid
                logger.info(f"Registered new feature '{feature_name}' version '{version}'.")
            
            conn.commit()
            conn.close()
            
            # Clear cache for this feature
            if feature_name in self.feature_cache:
                del self.feature_cache[feature_name]
            
            return feature_id
        
        except Exception as e:
            logger.error(f"Error registering feature: {str(e)}")
            raise
    
    def get_feature(self, feature_name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific feature and version.

        Args:
            feature_name: The name of the feature.
            version: The version of the feature.

        Returns:
            Feature metadata or None if not found.
        """
        try:
            # Check cache first
            cache_key = f"{feature_name}:{version}"
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
            
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # Handle "latest" version
            if version == "latest":
                cursor.execute(
                    """
                    SELECT id, name, version, description, data_type, statistics, 
                           tags, owner, status, created_at, updated_at
                    FROM features
                    WHERE name = ? AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (feature_name,)
                )
            else:
                cursor.execute(
                    """
                    SELECT id, name, version, description, data_type, statistics, 
                           tags, owner, status, created_at, updated_at
                    FROM features
                    WHERE name = ? AND version = ?
                    """,
                    (feature_name, version)
                )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                feature = {
                    'id': result[0],
                    'name': result[1],
                    'version': result[2],
                    'description': result[3],
                    'data_type': result[4],
                    'statistics': json.loads(result[5]) if result[5] else {},
                    'tags': json.loads(result[6]) if result[6] else [],
                    'owner': result[7],
                    'status': result[8],
                    'created_at': result[9],
                    'updated_at': result[10]
                }
                
                # Add lineage information
                feature['lineage'] = self.get_feature_lineage(feature['id'])
                
                # Add model usage information
                feature['model_usage'] = self.get_feature_model_usage(feature['id'])
                
                # Cache the result
                self.feature_cache[cache_key] = feature
                
                return feature
            
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving feature: {str(e)}")
            return None
    
    def list_features(self, 
                     feature_name: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     model_type: Optional[str] = None,
                     status: str = "active") -> List[Dict[str, Any]]:
        """
        List all registered features or versions of a specific feature.

        Args:
            feature_name: Optional name of a specific feature to list versions for.
            tags: Optional list of tags to filter features by.
            model_type: Optional model type to filter features used by specific models.
            status: Status of features to include (active, deprecated, all).

        Returns:
            A list of features and their metadata.
        """
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT id, name, version, description, data_type, statistics, 
                   tags, owner, status, created_at, updated_at
            FROM features
            """
            
            conditions = []
            params = []
            
            if feature_name:
                conditions.append("name = ?")
                params.append(feature_name)
            
            if status != "all":
                conditions.append("status = ?")
                params.append(status)
            
            if tags:
                # This is a simplistic approach; in a real system, you'd use a proper tag table
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")
                conditions.append("(" + " OR ".join(tag_conditions) + ")")
            
            if model_type:
                # Join with model_usage table
                query = query.replace("FROM features", """
                FROM features
                JOIN model_usage ON features.id = model_usage.feature_id
                """)
                conditions.append("model_usage.model_type = ?")
                params.append(model_type)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY name, created_at DESC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            features = []
            for result in results:
                feature = {
                    'id': result[0],
                    'name': result[1],
                    'version': result[2],
                    'description': result[3],
                    'data_type': result[4],
                    'statistics': json.loads(result[5]) if result[5] else {},
                    'tags': json.loads(result[6]) if result[6] else [],
                    'owner': result[7],
                    'status': result[8],
                    'created_at': result[9],
                    'updated_at': result[10]
                }
                features.append(feature)
            
            return features
        
        except Exception as e:
            logger.error(f"Error listing features: {str(e)}")
            return []
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics for feature data.
        
        Args:
            data: DataFrame with feature data
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        try:
            # Basic statistics
            if isinstance(data, pd.DataFrame):
                if len(data.columns) == 1:
                    # Single column DataFrame
                    col = data.columns[0]
                    series = data[col]
                    
                    if pd.api.types.is_numeric_dtype(series):
                        stats.update({
                            'min': float(series.min()),
                            'max': float(series.max()),
                            'mean': float(series.mean()),
                            'median': float(series.median()),
                            'std': float(series.std()),
                            'count': int(series.count()),
                            'missing': int(series.isna().sum())
                        })
                    elif pd.api.types.is_string_dtype(series):
                        stats.update({
                            'unique_count': int(series.nunique()),
                            'count': int(series.count()),
                            'missing': int(series.isna().sum())
                        })
                else:
                    # Multi-column DataFrame - calculate stats for each column
                    stats['columns'] = {}
                    for col in data.columns:
                        series = data[col]
                        col_stats = {}
                        
                        if pd.api.types.is_numeric_dtype(series):
                            col_stats.update({
                                'min': float(series.min()),
                                'max': float(series.max()),
                                'mean': float(series.mean()),
                                'median': float(series.median()),
                                'std': float(series.std()),
                                'count': int(series.count()),
                                'missing': int(series.isna().sum())
                            })
                        elif pd.api.types.is_string_dtype(series):
                            col_stats.update({
                                'unique_count': int(series.nunique()),
                                'count': int(series.count()),
                                'missing': int(series.isna().sum())
                            })
                        
                        stats['columns'][col] = col_stats
            else:
                # Single Series
                series = data
                
                if pd.api.types.is_numeric_dtype(series):
                    stats.update({
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'mean': float(series.mean()),
                        'median': float(series.median()),
                        'std': float(series.std()),
                        'count': int(series.count()),
                        'missing': int(series.isna().sum())
                    })
                elif pd.api.types.is_string_dtype(series):
                    stats.update({
                        'unique_count': int(series.nunique()),
                        'count': int(series.count()),
                        'missing': int(series.isna().sum())
                    })
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
        
        return stats
    
    def store_feature_data(self, 
                         feature_name: str, 
                         data: pd.DataFrame,
                         version: str = "latest",
                         overwrite: bool = False) -> bool:
        """
        Store feature data in the feature store.

        Args:
            feature_name: The name of the feature.
            data: DataFrame with feature data.
            version: The version of the feature.
            overwrite: Whether to overwrite existing data.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get feature metadata
            feature = self.get_feature(feature_name, version)
            if not feature:
                logger.error(f"Feature '{feature_name}' version '{version}' not found.")
                return False
            
            # Create directory for feature data
            feature_dir = os.path.join(self.feature_data_dir, feature_name)
            os.makedirs(feature_dir, exist_ok=True)
            
            # Define file path
            file_path = os.path.join(feature_dir, f"{version}.parquet")
            
            # Check if file exists and overwrite is False
            if os.path.exists(file_path) and not overwrite:
                logger.warning(f"Feature data file already exists and overwrite=False.")
                return False
            
            # Calculate statistics
            statistics = self._calculate_statistics(data)
            
            # Update feature metadata with statistics
            feature_metadata = feature.copy()
            feature_metadata['statistics'] = statistics
            self.register_feature(
                feature_name=feature_name,
                feature_metadata=feature_metadata,
                version=version,
                description=feature.get('description', ''),
                data_type=feature.get('data_type', 'float'),
                tags=feature.get('tags', []),
                owner=feature.get('owner', ''),
                status=feature.get('status', 'active')
            )
            
            # Store data
            data.to_parquet(file_path, index=True)
            logger.info(f"Stored data for feature '{feature_name}' version '{version}'.")
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing feature data: {str(e)}")
            return False
    
    def get_feature_data(self, 
                       feature_name: str, 
                       version: str = "latest",
                       time_range: Optional[Tuple[datetime, datetime]] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve feature data from the feature store.

        Args:
            feature_name: The name of the feature.
            version: The version of the feature.
            time_range: Optional time range to retrieve data for.

        Returns:
            DataFrame with feature data or None if not found.
        """
        try:
            # Get feature metadata
            feature = self.get_feature(feature_name, version)
            if not feature:
                logger.error(f"Feature '{feature_name}' version '{version}' not found.")
                return None
            
            # Define file path
            feature_dir = os.path.join(self.feature_data_dir, feature_name)
            file_path = os.path.join(feature_dir, f"{feature['version']}.parquet")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Feature data file not found: {file_path}")
                return None
            
            # Load data
            data = pd.read_parquet(file_path)
            
            # Filter by time range if provided
            if time_range and len(time_range) == 2:
                start_time, end_time = time_range
                if isinstance(data.index, pd.DatetimeIndex):
                    data = data[(data.index >= start_time) & (data.index <= end_time)]
                elif 'timestamp' in data.columns:
                    data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
            
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving feature data: {str(e)}")
            return None
    
    def log_feature_lineage(self, 
                          feature_name: str, 
                          version: str,
                          source_features: List[Dict[str, str]], 
                          transformation: str) -> bool:
        """
        Log the lineage of a derived feature.

        Args:
            feature_name: The name of the derived feature.
            version: The version of the derived feature.
            source_features: A list of dictionaries, each with 'name' and 'version' of source features.
            transformation: A description or code of the transformation applied.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get feature ID
            feature = self.get_feature(feature_name, version)
            if not feature:
                logger.error(f"Feature '{feature_name}' version '{version}' not found.")
                return False
            
            feature_id = feature['id']
            
            # Get current timestamp
            now = datetime.now().isoformat()
            
            # Connect to database
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # Insert lineage records
            for source in source_features:
                source_name = source.get('name')
                source_version = source.get('version', 'latest')
                
                # Get source feature ID
                source_feature = self.get_feature(source_name, source_version)
                if not source_feature:
                    logger.warning(f"Source feature '{source_name}' version '{source_version}' not found.")
                    continue
                
                source_id = source_feature['id']
                
                # Insert lineage record
                cursor.execute(
                    """
                    INSERT INTO lineage 
                    (feature_id, source_feature_id, transformation, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (feature_id, source_id, transformation, now)
                )
            
            conn.commit()
            conn.close()
            
            # Clear cache for this feature
            cache_key = f"{feature_name}:{version}"
            if cache_key in self.feature_cache:
                del self.feature_cache[cache_key]
            
            logger.info(f"Lineage logged for feature '{feature_name}' version '{version}'.")
            return True
        
        except Exception as e:
            logger.error(f"Error logging feature lineage: {str(e)}")
            return False
    
    def get_feature_lineage(self, feature_id: int) -> List[Dict[str, Any]]:
        """
        Get the lineage for a feature.

        Args:
            feature_id: The ID of the feature.

        Returns:
            List of lineage records.
        """
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT l.id, l.transformation, l.created_at,
                       sf.id, sf.name, sf.version
                FROM lineage l
                JOIN features sf ON l.source_feature_id = sf.id
                WHERE l.feature_id = ?
                ORDER BY l.created_at DESC
                """,
                (feature_id,)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            lineage = []
            for result in results:
                lineage_record = {
                    'id': result[0],
                    'transformation': result[1],
                    'created_at': result[2],
                    'source_feature': {
                        'id': result[3],
                        'name': result[4],
                        'version': result[5]
                    }
                }
                lineage.append(lineage_record)
            
            return lineage
        
        except Exception as e:
            logger.error(f"Error retrieving feature lineage: {str(e)}")
            return []
    
    def log_model_feature_usage(self, 
                              feature_name: str, 
                              version: str,
                              model_type: str,
                              model_name: str,
                              model_version: str = "latest",
                              importance: float = 0.0) -> bool:
        """
        Log the usage of a feature by a model.

        Args:
            feature_name: The name of the feature.
            version: The version of the feature.
            model_type: The type of model (e.g., 'classifier', 'regressor').
            model_name: The name of the model.
            model_version: The version of the model.
            importance: The importance of the feature in the model.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get feature ID
            feature = self.get_feature(feature_name, version)
            if not feature:
                logger.error(f"Feature '{feature_name}' version '{version}' not found.")
                return False
            
            feature_id = feature['id']
            
            # Get current timestamp
            now = datetime.now().isoformat()
            
            # Connect to database
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # Check if record already exists
            cursor.execute(
                """
                SELECT id FROM model_usage
                WHERE feature_id = ? AND model_type = ? AND model_name = ? AND model_version = ?
                """,
                (feature_id, model_type, model_name, model_version)
            )
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute(
                    """
                    UPDATE model_usage
                    SET importance = ?, created_at = ?
                    WHERE id = ?
                    """,
                    (importance, now, existing[0])
                )
            else:
                # Insert new record
                cursor.execute(
                    """
                    INSERT INTO model_usage
                    (feature_id, model_type, model_name, model_version, importance, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (feature_id, model_type, model_name, model_version, importance, now)
                )
            
            conn.commit()
            conn.close()
            
            # Clear cache for this feature
            cache_key = f"{feature_name}:{version}"
            if cache_key in self.feature_cache:
                del self.feature_cache[cache_key]
            
            logger.info(f"Model usage logged for feature '{feature_name}' version '{version}'.")
            return True
        
        except Exception as e:
            logger.error(f"Error logging model feature usage: {str(e)}")
            return False
    
    def get_feature_model_usage(self, feature_id: int) -> List[Dict[str, Any]]:
        """
        Get the models that use a feature.

        Args:
            feature_id: The ID of the feature.

        Returns:
            List of model usage records.
        """
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT id, model_type, model_name, model_version, importance, created_at
                FROM model_usage
                WHERE feature_id = ?
                ORDER BY created_at DESC
                """,
                (feature_id,)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            usage = []
            for result in results:
                usage_record = {
                    'id': result[0],
                    'model_type': result[1],
                    'model_name': result[2],
                    'model_version': result[3],
                    'importance': result[4],
                    'created_at': result[5]
                }
                usage.append(usage_record)
            
            return usage
        
        except Exception as e:
            logger.error(f"Error retrieving feature model usage: {str(e)}")
            return []
    
    def get_model_features(self, 
                         model_type: str,
                         model_name: str,
                         model_version: str = "latest") -> List[Dict[str, Any]]:
        """
        Get the features used by a model.

        Args:
            model_type: The type of model.
            model_name: The name of the model.
            model_version: The version of the model.

        Returns:
            List of features used by the model.
        """
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT f.id, f.name, f.version, f.description, f.data_type,
                       f.statistics, f.tags, f.owner, f.status, f.created_at, f.updated_at,
                       mu.importance
                FROM features f
                JOIN model_usage mu ON f.id = mu.feature_id
                WHERE mu.model_type = ? AND mu.model_name = ? AND mu.model_version = ?
                ORDER BY mu.importance DESC
                """,
                (model_type, model_name, model_version)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            features = []
            for result in results:
                feature = {
                    'id': result[0],
                    'name': result[1],
                    'version': result[2],
                    'description': result[3],
                    'data_type': result[4],
                    'statistics': json.loads(result[5]) if result[5] else {},
                    'tags': json.loads(result[6]) if result[6] else [],
                    'owner': result[7],
                    'status': result[8],
                    'created_at': result[9],
                    'updated_at': result[10],
                    'importance': result[11]
                }
                features.append(feature)
            
            return features
        
        except Exception as e:
            logger.error(f"Error retrieving model features: {str(e)}")
            return []
    
    def version_feature(self, 
                      feature_name: str, 
                      new_version: str, 
                      old_version: str = "latest") -> bool:
        """
        Create a new version of an existing feature.

        Args:
            feature_name: The name of the feature.
            new_version: The new version to create.
            old_version: The existing version to copy from.

        Returns:
            True if versioning was successful, False otherwise.
        """
        try:
            # Get old feature metadata
            old_feature = self.get_feature(feature_name, old_version)
            if not old_feature:
                logger.warning(f"Feature '{feature_name}' version '{old_version}' not found, cannot version.")
                return False
            
            # Create new version
            new_feature_id = self.register_feature(
                feature_name=feature_name,
                feature_metadata=old_feature,
                version=new_version,
                description=old_feature.get('description', ''),
                data_type=old_feature.get('data_type', 'float'),
                tags=old_feature.get('tags', []),
                owner=old_feature.get('owner', ''),
                status='active'
            )
            
            # Copy data if it exists
            old_data_path = os.path.join(self.feature_data_dir, feature_name, f"{old_feature['version']}.parquet")
            if os.path.exists(old_data_path):
                new_data_path = os.path.join(self.feature_data_dir, feature_name, f"{new_version}.parquet")
                
                # Load and save data
                data = pd.read_parquet(old_data_path)
                data.to_parquet(new_data_path, index=True)
            
            logger.info(f"Feature '{feature_name}' version '{old_version}' versioned to '{new_version}'.")
            return True
        
        except Exception as e:
            logger.error(f"Error versioning feature: {str(e)}")
            return False
    
    def search_features(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for features by name, description, or tags.

        Args:
            query: The search query.

        Returns:
            List of matching features.
        """
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # Search in name, description, and tags
            cursor.execute(
                """
                SELECT id, name, version, description, data_type, statistics, 
                       tags, owner, status, created_at, updated_at
                FROM features
                WHERE name LIKE ? OR description LIKE ? OR tags LIKE ?
                ORDER BY name, created_at DESC
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%")
            )
            
            results = cursor.fetchall()
            conn.close()
            
            features = []
            for result in results:
                feature = {
                    'id': result[0],
                    'name': result[1],
                    'version': result[2],
                    'description': result[3],
                    'data_type': result[4],
                    'statistics': json.loads(result[5]) if result[5] else {},
                    'tags': json.loads(result[6]) if result[6] else [],
                    'owner': result[7],
                    'status': result[8],
                    'created_at': result[9],
                    'updated_at': result[10]
                }
                features.append(feature)
            
            return features
        
        except Exception as e:
            logger.error(f"Error searching features: {str(e)}")
            return []
    
    def get_features_for_model_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all features compatible with a specific model type.

        Args:
            model_type: The type of model.

        Returns:
            List of compatible features.
        """
        try:
            # Get all features used by this model type
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT DISTINCT f.id, f.name, f.version
                FROM features f
                JOIN model_usage mu ON f.id = mu.feature_id
                WHERE mu.model_type = ? AND f.status = 'active'
                """,
                (model_type,)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            features = []
            for result in results:
                feature_id, feature_name, feature_version = result
                feature = self.get_feature(feature_name, feature_version)
                if feature:
                    features.append(feature)
            
            return features
        
        except Exception as e:
            logger.error(f"Error retrieving features for model type: {str(e)}")
            return []
    
    def integrate_with_mlflow(self, mlflow_run_id: str, feature_names: List[str]) -> bool:
        """
        Integrate feature store with MLflow by logging feature metadata to a run.

        Args:
            mlflow_run_id: The MLflow run ID.
            feature_names: List of feature names used in the run.

        Returns:
            True if successful, False otherwise.
        """
        try:
            import mlflow
            
            # Set active run
            with mlflow.start_run(run_id=mlflow_run_id):
                # Log feature info as params
                for feature_name in feature_names:
                    feature = self.get_feature(feature_name)
                    if feature:
                        # Log feature metadata
                        mlflow.log_param(f"feature_{feature_name}_version", feature.get('version', 'unknown'))
                        mlflow.log_param(f"feature_{feature_name}_type", feature.get('data_type', 'unknown'))
                        
                        # Log feature lineage
                        if 'lineage' in feature and feature['lineage']:
                            source_features = [src['source_feature']['name'] for src in feature['lineage']]
                            mlflow.log_param(f"feature_{feature_name}_sources", ','.join(source_features))
                        
                        # Log feature statistics as metrics if available
                        if 'statistics' in feature and feature['statistics']:
                            stats = feature['statistics']
                            for stat_name, stat_value in stats.items():
                                if isinstance(stat_value, (int, float)):
                                    mlflow.log_metric(f"feature_{feature_name}_{stat_name}", stat_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error integrating with MLflow: {str(e)}")
            return False
