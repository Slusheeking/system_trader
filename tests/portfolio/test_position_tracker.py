import json
import os

import pandas as pd
import pytest

from system_trader.portfolio.position_tracker import PositionTracker


tmp_positions_filename = "positions.json"

@pytest.fixture
def temp_positions_file(tmp_path):
    """
    Fixture for a temporary positions file path.
    """
    return tmp_path / tmp_positions_filename

@pytest.fixture(autouse=True)
def patch_config_load(monkeypatch, temp_positions_file):
    """
    Automatically patch load_config to return the temp positions file path.
    """
    monkeypatch.patch(
        'system_trader.portfolio.position_tracker.load_config',
        lambda: {'positions_file_path': str(temp_positions_file)}
    )
    return None


def test_load_positions_no_file(temp_positions_file):
    """
    When the positions file does not exist, load_positions should return an empty dict.
    """
    if temp_positions_file.exists():
        temp_positions_file.unlink()

    tracker = PositionTracker()
    positions = tracker.load_positions()
    assert positions == {}


def test_load_positions_valid_file(temp_positions_file):
    """
    When the positions file contains valid JSON mapping, load_positions should return it as floats.
    """
    data = {'AAPL': 10, 'GOOG': 5.5}
    temp_positions_file.write_text(json.dumps(data))

    tracker = PositionTracker()
    positions = tracker.load_positions()

    expected = {'AAPL': 10.0, 'GOOG': 5.5}
    assert positions == expected


def test_load_positions_invalid_json(temp_positions_file):
    """
    Invalid JSON should cause a JSONDecodeError.
    """
    temp_positions_file.write_text("not a valid json")

    tracker = PositionTracker()
    with pytest.raises(json.JSONDecodeError):
        tracker.load_positions()


def test_load_positions_wrong_type(temp_positions_file):
    """
    JSON content that is not a dict should cause a ValueError.
    """
    temp_positions_file.write_text(json.dumps(["AAPL", 10]))

    tracker = PositionTracker()
    with pytest.raises(ValueError):
        tracker.load_positions()


def test_track_positions_with_symbol_column():
    """
    DataFrame with 'symbol' and 'position' columns should map correctly.
    """
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOG'],
        'position': [10, 5]
    })
    tracker = PositionTracker()
    positions = tracker.track_positions(df)
    assert positions == {'AAPL': 10.0, 'GOOG': 5.0}


def test_track_positions_with_index_and_position():
    """
    DataFrame with index as symbol and 'position' column should map correctly.
    """
    df = pd.DataFrame({'position': [7, 3]}, index=['MSFT', 'TSLA'])
    tracker = PositionTracker()
    positions = tracker.track_positions(df)
    assert positions == {'MSFT': 7.0, 'TSLA': 3.0}


def test_track_positions_missing_position_column():
    """
    Missing 'position' column should cause a ValueError.
    """
    df = pd.DataFrame({'symbol': ['AAPL']})
    tracker = PositionTracker()
    with pytest.raises(ValueError):
        tracker.track_positions(df)


def test_track_positions_not_dataframe():
    """
    Non-DataFrame input should cause a TypeError.
    """
    tracker = PositionTracker()
    with pytest.raises(TypeError):
        tracker.track_positions("not a dataframe")


def test_save_positions_creates_file_and_content(temp_positions_file):
    """
    save_positions should create the file and write JSON mapping.
    """
    tracker = PositionTracker()
    data = {'AAPL': 12.5, 'GOOG': 3}
    tracker.save_positions(data)

    assert temp_positions_file.exists()
    text = temp_positions_file.read_text()
    loaded = json.loads(text)
    expected = {'AAPL': 12.5, 'GOOG': 3.0}
    assert loaded == expected


def test_save_positions_creates_nested_directory(tmp_path, monkeypatch):
    """
    save_positions should create nested directories if needed.
    """
    nested_path = tmp_path / "nested" / "dir" / tmp_positions_filename
    monkeypatch.patch(
        'system_trader.portfolio.position_tracker.load_config',
        lambda: {'positions_file_path': str(nested_path)}
    )
    tracker = PositionTracker()
    positions = {'TEST': 1}
    tracker.save_positions(positions)

    assert nested_path.exists()
    content = json.loads(nested_path.read_text())
    assert content == {'TEST': 1.0}
