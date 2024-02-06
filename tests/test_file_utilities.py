import os
import json
import pytest
from coastseg.file_utilities import read_json_file
from coastseg.file_utilities import to_file
from coastseg import file_utilities
import datetime
import numpy as np
from shapely.geometry import Point
import geopandas as gpd


def test_config_to_file_geodataframe_folder(tmp_path):
    gdf = gpd.GeoDataFrame({'geometry': [Point(1, 2)], 'value': ['test']})
    test_file = tmp_path / "config_gdf.geojson"
    file_utilities.config_to_file(gdf, str(tmp_path))

    loaded_gdf = gpd.read_file(test_file)
    assert isinstance(loaded_gdf, gpd.GeoDataFrame)
    assert loaded_gdf['value'][0] == 'test'

def test_config_to_file_geodataframe(tmp_path):
    gdf = gpd.GeoDataFrame({'geometry': [Point(1, 2)], 'value': ['test']})
    test_file = tmp_path / "config_gdf.geojson"
    file_utilities.config_to_file(gdf, str(test_file))

    loaded_gdf = gpd.read_file(test_file)
    assert isinstance(loaded_gdf, gpd.GeoDataFrame)
    assert loaded_gdf['value'][0] == 'test'

def test_config_to_file_json_folder(tmp_path):
    test_data = {"key": "value"}
    test_file = tmp_path / "config.json"
    file_utilities.config_to_file(test_data, str(tmp_path))

    with open(test_file, "r") as file:
        data = json.load(file)
        assert data == test_data    
    
def test_config_to_file_json(tmp_path):
    test_data = {"key": "value"}
    test_file = tmp_path / "config.json"
    file_utilities.config_to_file(test_data, str(test_file))

    with open(test_file, "r") as file:
        data = json.load(file)
        assert data == test_data

def test_write_to_json(tmp_path):
    test_data = {"key": "value"}
    test_file = tmp_path / "test.json"
    file_utilities.write_to_json(str(test_file), test_data)

    with open(test_file, "r") as file:
        data = json.load(file)
        assert data == test_data

def test_serialize_basic_dict(tmp_path):
    data = {"key1": "value1", "key2": 123}
    file_path = tmp_path / "test_basic.json"
    to_file(data, str(file_path))

    with open(file_path, "r") as file:
        loaded_data = json.load(file)
        assert loaded_data == data

def test_serialize_datetime(tmp_path):
    date_obj = datetime.date.today()
    datetime_obj = datetime.datetime.now()
    data = {"date": date_obj, "datetime": datetime_obj}
    file_path = tmp_path / "test_datetime.json"
    to_file(data, str(file_path))

    with open(file_path, "r") as file:
        loaded_data = json.load(file)
        assert loaded_data == {"date": date_obj.isoformat(), "datetime": datetime_obj.isoformat()}

def test_serialize_numpy_array(tmp_path):
    np_array = np.array([1, 2, 3])
    data = {"array": np_array}
    file_path = tmp_path / "test_numpy.json"
    to_file(data, str(file_path))

    with open(file_path, "r") as file:
        loaded_data = json.load(file)
        assert loaded_data == {"array": np_array.tolist()}


def test_move_from_dir_to_dir(temp_src_dir, temp_dst_dir):
    """Test moving all files from one directory to another"""
    file_utilities.move_files(temp_src_dir, temp_dst_dir)
    assert len(os.listdir(temp_dst_dir)) == 5
    assert len(os.listdir(temp_src_dir)) == 0  # Source dir should be empty
    assert os.path.exists(temp_src_dir)


def test_move_from_file_list_to_dir(temp_src_files, temp_dst_dir):
    """Test moving a list of files to a directory"""
    file_utilities.move_files(temp_src_files, temp_dst_dir)
    assert len(os.listdir(temp_dst_dir)) == 5
    # Check if source files are deleted
    for file_path in temp_src_files:
        assert not os.path.exists(file_path)


def test_move_from_dir_path_to_dir(temp_src_dir, temp_dst_dir):
    """Test moving all files from a source directory path to a destination directory"""
    file_utilities.move_files(str(temp_src_dir), temp_dst_dir)
    assert (
        len(os.listdir(temp_dst_dir)) == 5
    ), "All files should be moved to the destination directory"
    assert (
        len(os.listdir(temp_src_dir)) == 0
    ), "Source directory should be empty after moving files"


def test_delete_source_directory(temp_src_dir, temp_dst_dir):
    """Test deleting the source directory after moving files"""
    file_utilities.move_files(temp_src_dir, temp_dst_dir, delete_src=True)
    assert len(os.listdir(temp_dst_dir)) == 5
    assert not os.path.exists(temp_src_dir)  # Source dir should be deleted


def test_delete_source_files(temp_src_files, temp_dst_dir):
    """Test deleting source files after moving them"""
    file_utilities.move_files(temp_src_files, temp_dst_dir, delete_src=True)
    assert len(os.listdir(temp_dst_dir)) == 5
    for file_path in temp_src_files:
        assert not os.path.exists(file_path)  # Source files should be deleted


def test_read_existing_json_file(tmpdir):
    # Create a temporary JSON file
    json_data = {"key": "value"}
    json_file = tmpdir.join("test.json")
    json_file.write(json.dumps(json_data))

    # Read the JSON file
    result = read_json_file(str(json_file))

    # Check that the result matches the expected data
    assert result == json_data


def test_read_non_existing_json_file(tmpdir):
    # Create a temporary directory
    directory = tmpdir.mkdir("test_directory")

    # Attempt to read a non-existing JSON file
    with pytest.raises(FileNotFoundError):
        read_json_file(str(directory.join("non_existing.json")), raise_error=True)


def test_read_non_existing_json_file_no_error(tmpdir):
    # Create a temporary directory
    directory = tmpdir.mkdir("test_directory")

    # Attempt to read a non-existing JSON file without raising an error
    result = read_json_file(str(directory.join("non_existing.json")), raise_error=False)

    # Check that an empty dictionary is returned
    assert result == {}


def test_read_invalid_json_file(tmpdir):
    # Create a temporary JSON file with invalid JSON data
    invalid_json = "not a valid JSON"
    json_file = tmpdir.join("test.json")
    json_file.write(invalid_json)

    # Attempt to read the invalid JSON file
    with pytest.raises(json.JSONDecodeError):
        read_json_file(str(json_file), raise_error=True)
