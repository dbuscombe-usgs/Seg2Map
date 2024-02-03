import json

from coastseg import shoreline
from coastseg import transects
from coastseg import roi
from coastseg import exceptions
from coastseg import file_utilities
from coastseg import coastseg_map
from coastseg import common
from leafmap import Map
import pytest
import geopandas as gpd
from ipyleaflet import GeoJSON
from shapely import geometry

# def test_set_roi_settings():


def test_imports():
    """Test that all the internal coastseg packages are imported correctly"""
    from coastseg import exceptions
    from coastseg import exception_handler
    from coastseg import extracted_shoreline
    from coastseg import factory
    from coastseg import map_UI
    from coastseg import models_UI


def test_save_config_invalid_inputs(
    valid_coastseg_map,
    valid_coastseg_map_with_incomplete_settings,
    valid_coastseg_map_with_settings,
    coastseg_map_with_rois,
    valid_rois_filepath,
):
    with pytest.raises(Exception):
        valid_coastseg_map.save_config()

    # test if exception is raised when settings is missing ["dates", "sat_list", "landsat_collection"]
    # save config will not work without ROIs loaded onto map
    valid_coastseg_map_with_incomplete_settings.load_feature_on_map(
        "rois", file=valid_rois_filepath
    )

    with pytest.raises(Exception):
        valid_coastseg_map_with_incomplete_settings.save_config()

    # test if exception is raised when coastseg_map missing rois
    with pytest.raises(Exception):
        # save config will not work without ROIs loaded onto map
        valid_coastseg_map_with_settings.save_config()

    # test if exception is raised when coastseg_map missing selected_layer
    with pytest.raises(Exception):
        coastseg_map_with_rois.save_config()


def test_save_config(coastseg_map_with_selected_roi_layer, tmp_path):
    """tests if save configs will save both a config.json and
    config_gdf.geojson to the filepath directory when coastseg_map's rois have roi_settings.
    Args:
        coastseg_map_with_selected_roi_layer (Coastseg_Map): instance of CoastSeg_Map with settings loaded, rois loaded,
                                                        and ROI with id 17 selected on map
                                                        ROIs on map have ids:["17","30","35"]
    Selected ROIs have id:["17"]
        tmp_path (WindowsPath): temporary directory
    """
    actual_coastsegmap = coastseg_map_with_selected_roi_layer
    filepath = str(tmp_path)
    roi_id = "17"
    date_str = "01-31-22_12_19_45"

    # modify the settings
    settings = actual_coastsegmap.get_settings()
    dates = settings["dates"]
    landsat_collection = settings["landsat_collection"]
    sat_list = settings["sat_list"]

    # Add roi_settings to  actual_coastsegmap.rois
    selected_layer = actual_coastsegmap.map.find_layer(roi.ROI.SELECTED_LAYER_NAME)
    roi_settings = common.create_roi_settings(
        settings,
        selected_layer.data,
        filepath,
        date_str,
    )
    actual_coastsegmap.rois.set_roi_settings(roi_settings)
    assert actual_coastsegmap.rois.roi_settings != {}
    # use the roi_settings to save config
    actual_coastsegmap.save_config(filepath)
    assert actual_coastsegmap.rois.roi_settings != {}
    expected_config_json_path = tmp_path / "config.json"
    assert expected_config_json_path.exists()
    with open(expected_config_json_path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    # test if roi id was saved as key and key fields exist
    assert roi_id in data
    assert "dates" in data[roi_id]
    assert dates == data[roi_id]["dates"]
    assert "sat_list" in data[roi_id]
    assert sat_list == data[roi_id]["sat_list"]
    assert "roi_id" in data[roi_id]
    assert roi_id == data[roi_id]["roi_id"]
    assert "polygon" in data[roi_id]
    assert "landsat_collection" in data[roi_id]
    assert landsat_collection in data[roi_id]["landsat_collection"]
    assert "sitename" in data[roi_id]
    assert date_str in data[roi_id]["sitename"]
    assert "filepath" in data[roi_id]
    assert filepath == data[roi_id]["filepath"]
    expected_config_geojson_path = tmp_path / "config_gdf.geojson"
    assert expected_config_geojson_path.exists()


def test_save_config_empty_roi_settings(coastseg_map_with_selected_roi_layer, tmp_path):
    """test_save_config_empty_roi_settings tests if save configs will save both a config.json and
    config_gdf.geojson to the filepath directory when coastseg_map's rois do not have roi_settings.
    It should also create roi_settings for coastseg_map's rois

    Args:
        coastseg_map_with_selected_roi_layer (Coastseg_Map): instance of CoastSeg_Map with settings loaded, rois loaded,
                                                        and ROI with id 17 selected on map
                                                        ROIs on map have ids:["17","30","35"]
    Selected ROIs have id:["17"]
        tmp_path (WindowsPath): temporary directory
    """
    actual_coastsegmap = coastseg_map_with_selected_roi_layer
    assert actual_coastsegmap.rois.roi_settings == {}
    filepath = str(tmp_path)
    roi_id = "17"
    actual_coastsegmap.save_config(filepath)
    # roi_settings was empty before. save_config should have created it
    assert actual_coastsegmap.rois.roi_settings != {}
    expected_config_json_path = tmp_path / "config.json"
    assert expected_config_json_path.exists()
    with open(expected_config_json_path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    # test if roi id was saved as key and key fields exist
    assert roi_id in data
    assert "dates" in data[roi_id]
    assert "sat_list" in data[roi_id]
    assert "roi_id" in data[roi_id]
    assert "polygon" in data[roi_id]
    assert "landsat_collection" in data[roi_id]
    assert "sitename" in data[roi_id]
    assert "filepath" in data[roi_id]
    expected_config_geojson_path = tmp_path / "config_gdf.geojson"
    assert expected_config_geojson_path.exists()


def test_load_json_config_without_rois(valid_coastseg_map_with_settings, tmp_data_path):
    # test if exception is raised when coastseg_map has no ROIs
    actual_coastsegmap = valid_coastseg_map_with_settings
    with pytest.raises(Exception):
        actual_coastsegmap.load_json_config("")


def test_load_json_config_downloaded(
    valid_coastseg_map_with_settings,
    valid_rois_filepath,
    config_json,
):
    config_path, temp_dir = config_json
    # tests if load_json_config will load contents into rois.roi_settings
    # create instance of Coastseg_Map with settings and ROIs initially loaded
    actual_coastsegmap = valid_coastseg_map_with_settings
    actual_coastsegmap.load_feature_on_map("rois", file=valid_rois_filepath)

    # test if settings are correctly loaded when valid json config loaded with 'filepath' & 'sitename' keys is loaded
    json_data = actual_coastsegmap.load_json_config(config_path)
    actual_coastsegmap.rois.roi_settings = common.process_roi_settings(json_data, temp_dir)
    
    assert isinstance(actual_coastsegmap.rois.roi_settings, dict)
    actual_config = file_utilities.read_json_file(config_path)
    for roi_id in actual_config["roi_ids"]:
        assert roi_id in actual_coastsegmap.rois.roi_settings
        for key in actual_config[roi_id]:
            assert (
                actual_coastsegmap.rois.roi_settings[roi_id][key]
                == actual_config[roi_id][key]
            )
    for roi_id, item in actual_config.get("settings", {}).items():
        assert actual_coastsegmap.settings[roi_id] == item


# I don't think this can even happen in the current version of coastseg
# this test is a relic from when coastseg had a save_config button that allowed users to save
# not downloaded sessions.
# def test_load_json_config_when_data_path_not_exist(
#     valid_coastseg_map_with_settings,
#     valid_rois_filepath,
#     config_json_no_sitename_dir,
# ):
#     config_path, temp_dir = config_json_no_sitename_dir
#     # tests if load_json_config will load contents into rois.roi_settings when rois have not been downloaded before
#     # create instance of Coastseg_Map with settings and ROIs initially loaded
#     actual_coastsegmap = valid_coastseg_map_with_settings
#     actual_coastsegmap.load_feature_on_map("rois", file=valid_rois_filepath)
#     # test if settings are correctly loaded when valid json config without 'filepath' & 'sitename' keys is loaded

#     json_data = actual_coastsegmap.load_json_config(config_path)
#     actual_coastsegmap.roi_settings = common.process_roi_settings(json_data, temp_dir)
#     with pytest.raises(exceptions.WarningException):
            

def test_valid_shoreline_gdf(valid_shoreline_gdf: gpd.GeoDataFrame):
    """tests if a Shoreline will be created from a valid shoreline thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): a valid shoreline as a gpd.GeoDataFrame
    """
    expected_shoreline = shoreline.Shoreline(shoreline=valid_shoreline_gdf)
    assert isinstance(expected_shoreline, shoreline.Shoreline)
    assert expected_shoreline.gdf is not None
    assert expected_shoreline.filename == "shoreline.geojson"


def test_valid_transects_gdf(valid_transects_gdf: gpd.GeoDataFrame):
    """tests if a Transects will be created from a valid transects thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): valid transects as a gpd.GeoDataFrame
    """
    expected_transects = transects.Transects(transects=valid_transects_gdf)
    assert isinstance(expected_transects, transects.Transects)
    assert expected_transects.gdf is not None
    assert expected_transects.filename == "transects.geojson"


def test_transect_compatible_roi(transect_compatible_roi: gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    expected_roi = roi.ROI(rois_gdf=transect_compatible_roi)
    assert isinstance(expected_roi, roi.ROI)
    assert expected_roi.gdf is not None
    assert expected_roi.filename == "rois.geojson"


def test_transect_compatible_roi(transect_compatible_roi: gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    expected_roi = roi.ROI(rois_gdf=transect_compatible_roi)
    assert isinstance(expected_roi, roi.ROI)
    assert expected_roi.gdf is not None
    assert expected_roi.filename == "rois.geojson"


def test_coastseg_map():
    """tests a CoastSeg_Map object is created"""
    coastsegmap = coastseg_map.CoastSeg_Map()
    assert isinstance(coastsegmap, coastseg_map.CoastSeg_Map)
    assert isinstance(coastsegmap.map, Map)
    assert hasattr(coastsegmap, "draw_control")
    assert hasattr(coastsegmap, "settings")
    default_settings = {
        "landsat_collection": "C02",
        "dates": ["2017-12-01", "2018-01-01"],
        "sat_list": ["L8"],
        "cloud_thresh": 0.5,
        "dist_clouds": 300,
        "output_epsg": 4326,
        "check_detection": False,
        "adjust_detection": False,
        "save_figure": True,
        "min_beach_area": 4500,
        "min_length_sl": 100,
        "cloud_mask_issue": False,
        "sand_color": "default",
        "pan_off": "False",
        "max_dist_ref": 25,
        "along_dist": 25,
        "min_points": 3,
        "max_std": 15,
        "max_range": 30,
        "min_chainage": -100,
        "multiple_inter": "auto",
        "prc_multiple": 0.1,
        "apply_cloud_mask": True,
        "image_size_filter": True,
    }
    for key in default_settings:
        assert key in coastsegmap.settings
        assert coastsegmap.settings[key] == default_settings[key]


def test_set_settings():
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    coastsegmap = coastseg_map.CoastSeg_Map()
    pre_process_settings = {
        # general parameters:
        "dates": ["2018-12-01", "2019-03-01"],
        "sat_list": ["L9"],
        "cloud_thresh": 0.9,  # threshold on maximum cloud cover
        "dist_clouds": 400,  # ditance around clouds where shoreline can't be mapped
        "output_epsg": 3857,  # epsg code of spatial reference system desired for the output
        # quality control:
        "check_detection": True,  # if True, shows each shoreline detection to the user for validation
        "adjust_detection": False,  # if True, allows user to adjust the position of each shoreline by changing the threshold
        "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
        # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
        "min_beach_area": 400,  # minimum area (in metres^2) for an object to be labelled as a beach
        "min_length_sl": 100,  # minimum length (in metres) of shoreline perimeter to be valid
        "cloud_mask_issue": True,  # switch this parameter to True if sand pixels are masked (in black) on many images
        "sand_color": "default",  # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        "pan_off": "False",  # if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
        "max_dist_ref": 20,
        "landsat_collection": "C02",
    }
    coastsegmap.set_settings(**pre_process_settings)
    actual_settings = set(list(coastsegmap.get_settings().keys()))
    expected_settings = set(list(pre_process_settings.keys()))
    assert expected_settings.issubset(actual_settings)
    assert set(["dates", "landsat_collection", "sat_list"]).issubset(actual_settings)
    for key in pre_process_settings:
        assert coastsegmap.settings[key] == pre_process_settings[key]


def test_select_roi_layer(
    valid_coastseg_map_with_settings,
    valid_rois_filepath,
):
    """tests if a ROI will be added to selected layer when clicked
    Simulates an ROI being clicked on map by manually adding ROI id to selected_set
    and creating a new layer
    Args:
        valid_coastseg_map_with_settings (Coastseg_Map): valid instance of coastseg map with settings already loaded
        valid_rois_filepath (str): filepath to geojson file containing valid rois
                                    ROIs with ids:[17,30,35]
    """
    actual_coastsegmap = valid_coastseg_map_with_settings
    # test if rois will added to coastsegmap and added to ROI layer
    actual_coastsegmap.load_feature_on_map("rois", file=valid_rois_filepath)
    # test if roi layer was added to map
    existing_layer = actual_coastsegmap.map.find_layer(roi.ROI.LAYER_NAME)
    assert existing_layer is not None
    # simulate an ROI being clicked on map
    ROI_id = "17"
    actual_coastsegmap.selected_set.add(ROI_id)

    selected_layer = GeoJSON(
        data=actual_coastsegmap.convert_selected_set_to_geojson(
            actual_coastsegmap.selected_set, layer_name=roi.ROI.LAYER_NAME
        ),
        name=roi.ROI.SELECTED_LAYER_NAME,
        hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
    )
    actual_coastsegmap.replace_layer_by_name(
        roi.ROI.SELECTED_LAYER_NAME,
        selected_layer,
        on_click=actual_coastsegmap.selected_onclick_handler,
        on_hover=actual_coastsegmap.update_roi_html,
    )
    # test if roi layer was added to map
    selected_layer = actual_coastsegmap.map.find_layer(roi.ROI.SELECTED_LAYER_NAME)
    assert selected_layer is not None
    existing_layer = actual_coastsegmap.map.find_layer("Selected ROIs")
    assert existing_layer is not None
    assert "17" in actual_coastsegmap.selected_set
    assert isinstance(selected_layer.data, dict)
    assert isinstance(selected_layer.data["features"], list)
    assert isinstance(selected_layer.data["features"][0], dict)
    assert len(selected_layer.data["features"]) == 1
    roi_json = actual_coastsegmap.rois.gdf[
        actual_coastsegmap.rois.gdf["id"] == ROI_id
    ].to_json()
    roi_geojson = json.loads(roi_json)
    assert isinstance(roi_geojson, dict)
    # test if geojson in selected layer matches geojson in coastsegmap.rois.gdf
    assert (
        roi_geojson["features"][0]["geometry"]
        == selected_layer.data["features"][0]["geometry"]
    )


def test_load_rois_on_map_with_file(
    valid_coastseg_map_with_settings, valid_rois_filepath, valid_rois_gdf
):
    """tests if a ROI will be created from geojson file and added to the map
    Args:
        valid_coastseg_map_with_settings (Coastseg_Map): valid instance of coastseg map with settings already loaded
        valid_rois_filepath (str): filepath to geojson file containing valid rois
    """
    actual_coastsegmap = valid_coastseg_map_with_settings
    # test if rois will be correctly loaded onto map
    actual_coastsegmap.load_feature_on_map("rois", file=valid_rois_filepath)
    assert actual_coastsegmap.rois is not None
    assert isinstance(actual_coastsegmap.rois, roi.ROI)
    # test if rois geodataframe was created correctly
    assert isinstance(actual_coastsegmap.rois.gdf, gpd.GeoDataFrame)
    assert actual_coastsegmap.rois.gdf.equals(valid_rois_gdf)
    # test if roi layer was added to map
    existing_layer = actual_coastsegmap.map.find_layer(roi.ROI.LAYER_NAME)
    assert existing_layer is not None
