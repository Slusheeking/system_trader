import pytest
from system_trader.feature_store.feature_store import FeatureStore


def pytest_configure(config):
    # Optional: configure logging if needed
    pass


@ pytest.fixture
    def feature_store(tmp_path):
        # Use in-memory SQLite for SQLAlchemy
        config = {
            'connection_string': 'sqlite:///:memory:',
            'storage_dir': str(tmp_path / 'fs_data'),
        }
        fs = FeatureStore(config)
        return fs


def test_register_and_get_feature(feature_store):
    fs = feature_store
    # Register a new feature
    metadata = {'statistics': {'mean': 0.0, 'std': 1.0}}
    feature_id = fs.register_feature(
        feature_name='feat1',
        feature_metadata=metadata,
        version='v1',
        description='desc1',
        data_type='int',
        tags=['tag1', 'tag2'],
        owner='owner1',
        status='active'
    )
    # Retrieve via get_feature
    feat = fs.get_feature('feat1', 'v1')
    assert feat['id'] == feature_id
    assert feat['name'] == 'feat1'
    assert feat['version'] == 'v1'
    assert feat['description'] == 'desc1'
    assert feat['data_type'] == 'int'
    assert feat['statistics'] == metadata['statistics']
    assert feat['tags'] == ['tag1', 'tag2']
    assert feat['owner'] == 'owner1'
    assert feat['status'] == 'active'

    # List all features and verify the registered one exists
    all_feats = fs.list_features()
    assert any(
        f['name'] == 'feat1' and f['version'] == 'v1' and f['id'] == feature_id
        for f in all_feats
    )


def test_version_feature_creates_new_version(feature_store):
    fs = feature_store
    # Register initial version
    id1 = fs.register_feature('featX', {}, version='v1')
    orig = fs.get_feature('featX', 'v1')
    assert orig['id'] == id1

    # Create a new version based on v1
    result = fs.version_feature('featX', 'v2', old_version='v1')
    assert result is True

    # Verify new version exists and has a different id
    v2 = fs.get_feature('featX', 'v2')
    assert v2['version'] == 'v2'
    assert v2['id'] != id1

    # Ensure both versions appear in list_features
    feats = fs.list_features(feature_name='featX')
    versions = {f['version'] for f in feats if f['name'] == 'featX'}
    assert 'v1' in versions and 'v2' in versions


def test_log_and_get_lineage(feature_store):
    fs = feature_store
    # Register source features
    id_a = fs.register_feature('s1', {}, version='v1')
    id_b = fs.register_feature('s2', {}, version='v1')

    # Register derived feature
    id_d = fs.register_feature('derived', {}, version='v1')

    # Log lineage for the derived feature
    sources = [
        {'name': 's1', 'version': 'v1'},
        {'name': 's2', 'version': 'v1'}
    ]
    transformation = 's1 + s2'
    result = fs.log_feature_lineage('derived', 'v1', sources, transformation)
    assert result is True

    # Retrieve lineage by feature id
    lineage_records = fs.get_feature_lineage(id_d)
    assert len(lineage_records) == 1
    entry = lineage_records[0]
    assert entry['source_features'] == sources
    assert entry['transformation'] == transformation
