import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()

class FeatureVersion(Base):
    __tablename__ = 'feature_versions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class FeatureLineage(Base):
    __tablename__ = 'feature_lineages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    source_features = Column(JSON, nullable=False)
    transformation = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

class StorageManager:
    """
    Manages database connection and provides CRUD operations
    for FeatureVersion and FeatureLineage.
    """
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_feature_version(self, feature_name: str, version: str,
                               metadata: Dict[str, Any]) -> FeatureVersion:
        session: Session = self.Session()
        fv = FeatureVersion(
            feature_name=feature_name,
            version=version,
            metadata=metadata,
            created_at=datetime.utcnow()
        )
        session.add(fv)
        session.commit()
        session.refresh(fv)
        session.close()
        return fv

    def get_feature_version(self, feature_name: str,
                             version: str) -> Optional[FeatureVersion]:
        session: Session = self.Session()
        fv = session.query(FeatureVersion)
                     .filter_by(feature_name=feature_name, version=version)
                     .first()
        session.close()
        return fv

    def list_feature_versions(self, feature_name: str) -> List[FeatureVersion]:
        session: Session = self.Session()
        fvs = session.query(FeatureVersion)
                     .filter_by(feature_name=feature_name)
                     .order_by(FeatureVersion.created_at.desc())
                     .all()
        session.close()
        return fvs

    def create_lineage_entry(self, feature_name: str, version: str,
                              source_features: List[Dict[str, str]],
                              transformation: str) -> FeatureLineage:
        session: Session = self.Session()
        fl = FeatureLineage(
            feature_name=feature_name,
            version=version,
            source_features=source_features,
            transformation=transformation,
            timestamp=datetime.utcnow()
        )
        session.add(fl)
        session.commit()
        session.refresh(fl)
        session.close()
        return fl
