import os
import json
import tempfile
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon
from coastseg.tide_correction import save_transect_settings, get_seaward_points_gdf
from coastseg.tide_correction import load_regions_from_geojson
import pandas as pd
from coastseg.tide_correction import get_tide_predictions
from unittest.mock import patch
import numpy as np


def test_get_seaward_points_gdf():
    # Create a GeoDataFrame with transect data
    transects = gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "geometry": [
                LineString([(-75.19473124155853,
                38.13686333982983), (-75.16075424076779,
                38.12447790470557)]),
                LineString([( -75.20301831492232,
                38.12317405244161), ( -75.16862696046286,
                38.11274239609162)]),
            ],
        },
        crs = 4326
    ) 

    # Call the function to get the seaward points GeoDataFrame
    seaward_points_gdf = get_seaward_points_gdf(transects)

    # Check that the seaward points GeoDataFrame was created correctly
    assert isinstance(seaward_points_gdf, gpd.GeoDataFrame)
    assert len(seaward_points_gdf) == 2
    assert seaward_points_gdf.crs == "epsg:4326"
    assert seaward_points_gdf.columns.tolist() == ["transect_id", "geometry"]

    # Check the geometry of the seaward points
    assert seaward_points_gdf.loc[0, "geometry"] == Point(-75.16075424076779,38.12447790470557)
    assert seaward_points_gdf.loc[1, "geometry"] == Point( -75.16862696046286,
            38.11274239609162)

def test_get_seaward_points_gdf_diff_crs():
    # Create a GeoDataFrame with transect data
    transects = gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "geometry": [
                LineString([(-8370639.1921473555, 4598778.094918826), (-8366856.889720647, 4597025.320623461)]),
                LineString([( -8371561.704934379, 4596840.818066094), ( -8367733.276868261, 4595364.797606845)]),
            ],
        },
        crs = 3857
    ) 

    # Call the function to get the seaward points GeoDataFrame
    seaward_points_gdf = get_seaward_points_gdf(transects)

    # Check that the seaward points GeoDataFrame was created correctly
    assert isinstance(seaward_points_gdf, gpd.GeoDataFrame)
    assert len(seaward_points_gdf) == 2
    assert seaward_points_gdf.crs == "epsg:4326"
    assert seaward_points_gdf.columns.tolist() == ["transect_id", "geometry"]

    transects.to_crs(epsg=4326, inplace=True)
    assert list(seaward_points_gdf.loc[0, "geometry"].coords)[0] == list(transects.loc[0, "geometry"].coords)[1]
    # assert seaward_points_gdf.loc[1, "geometry"] == Point( -75.16862696046286,
    #         38.11274239609162)
 
def test_get_seaward_points_gdf_no_crs():
    # Create a GeoDataFrame with transect data
    transects = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)]),
            ],
        }
    )

    # Call the function to get the seaward points GeoDataFrame
    seaward_points_gdf = get_seaward_points_gdf(transects)

    # Check that the seaward points GeoDataFrame was created correctly
    assert isinstance(seaward_points_gdf, gpd.GeoDataFrame)
    assert len(seaward_points_gdf) == 3
    assert seaward_points_gdf.crs == "epsg:4326"
    assert seaward_points_gdf.columns.tolist() == ["transect_id", "geometry"]

    # Check the geometry of the seaward points
    assert seaward_points_gdf.loc[0, "geometry"] == Point(1, 1)
    assert seaward_points_gdf.loc[1, "geometry"] == Point(2, 2)
    assert seaward_points_gdf.loc[2, "geometry"] == Point(3, 3)
    

def test_load_regions_from_geojson():
    # Create a temporary GeoJSON file
    with tempfile.NamedTemporaryFile(suffix=".geojson") as tmp:
        geojson_path = tmp.name

        # Create a GeoDataFrame with some test data
        regions_gdf = gpd.GeoDataFrame(
            {
                'region_id': [1, 2, 3],
                'geometry': [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ]
            }
        )

        # Save the GeoDataFrame to the temporary GeoJSON file
        regions_gdf.to_file(geojson_path, driver='GeoJSON')

        # Call the function to load the regions
        loaded_regions_gdf = load_regions_from_geojson(geojson_path)

        # Check that the loaded regions is a GeoDataFrame
        assert isinstance(loaded_regions_gdf, gpd.GeoDataFrame)

        # Check that the 'region_id' column is added
        assert 'region_id' in loaded_regions_gdf.columns

        # Check that the number of regions is correct
        assert len(loaded_regions_gdf) == 3
        
        
def test_get_tide_predictions():
    # Test data setup
    x, y = 1.0, 2.0
    timeseries_df = pd.DataFrame({
        "dates": pd.date_range("2021-01-01", periods=3),
        "transect1": [0.5, 0.6, 0.7]
    })
    model_region_directory = "path/to/model/region"
    # invalid transect ID
    transect_id = "transect3"

    # Call the function
    # Mock the model_tides function for invalid transect ID
    with patch('coastseg.tide_correction.model_tides') as mock_model_tides:
        mock_model_tides.return_value = pd.DataFrame({"tide": [1.0, 2.0, 3.0],
                                                      "transect_id": np.repeat(transect_id,3),
                                                      "dates":timeseries_df.dates,
                                                      "x": np.repeat(x, len(timeseries_df)),
                                                      "y": np.repeat(y, len(timeseries_df))})
        result = get_tide_predictions(x, y, timeseries_df, model_region_directory, transect_id)
        assert result is None
        
    # Mock the model_tides function with no transect ID
    with patch('coastseg.tide_correction.model_tides') as mock_model_tides:
        mock_model_tides.return_value = pd.DataFrame({"tide": [1.0, 2.0, 3.0],
                                                      "dates":timeseries_df.dates,
                                                      "x": np.repeat(x, len(timeseries_df)),
                                                      "y": np.repeat(y, len(timeseries_df))})
        result = get_tide_predictions(x, y, timeseries_df, model_region_directory, "")
        # Assertions
        assert 'tide' in result.columns, "Result should contain 'tide' column"
        assert 'transect_id' not in result.columns, "Result should contain 'transect id' column"
        assert 'dates' in result.columns, "Result should contain 'dates' column"
        assert 'x' in result.columns, "Result should contain 'x' column"
        assert 'y' in result.columns, "Result should contain 'y' column"
        
        assert len(result) == 3, "Result should have 3 rows of predictions"
        assert mock_model_tides.called, "model_tides function should be called"
        assert (result['x'] == x).all(), "All 'x' values should be equal to input x"
        assert (result['y'] == y).all(), "All 'y' values should be equal to input y"

def test_get_tide_predictions_valid_transect():
    # Test data setup
    x, y = 1.0, 2.0
    timeseries_df = pd.DataFrame({
        "dates": pd.date_range("2021-01-01", periods=3),
        "transect1": [0.5, 0.6, 0.7]
    })
    model_region_directory = "path/to/model/region"
    transect_id = "transect1"

    # Call the function
    # Mock the model_tides function
    with patch('coastseg.tide_correction.model_tides') as mock_model_tides:
        mock_model_tides.return_value = pd.DataFrame({"tide": [1.0, 2.0, 3.0],
                                                      "transect_id": np.repeat(transect_id,3),
                                                      "dates":timeseries_df.dates,
                                                      "x": np.repeat(x, len(timeseries_df)),
                                                      "y": np.repeat(y, len(timeseries_df))})
        result = get_tide_predictions(x, y, timeseries_df, model_region_directory, transect_id)
        # Assertions
        assert 'tide' in result.columns, "Result should contain 'tide' column"
        assert 'transect_id' in result.columns, "Result should contain 'transect id' column"
        assert 'dates' in result.columns, "Result should contain 'dates' column"
        assert 'x' in result.columns, "Result should contain 'x' column"
        assert 'y' in result.columns, "Result should contain 'y' column"
        
        assert len(result) == 3, "Result should have 3 rows of predictions"
        assert mock_model_tides.called, "model_tides function should be called"
        assert (result['x'] == x).all(), "All 'x' values should be equal to input x"
        assert (result['y'] == y).all(), "All 'y' values should be equal to input y"
        assert (result['transect_id'] == transect_id).all(), "All 'transect_id' values should be equal to input transect_id"
