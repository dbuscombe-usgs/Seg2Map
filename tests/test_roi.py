import math
import pytest
from coastseg import roi
from coastseg import exceptions
import geopandas as gpd
import pyproj
from shapely.geometry import Polygon, LineString


# Test that when ROI's area is too large an error is thrown
def test_ROI_too_large():
    rectangle = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (-122.66944064253451, 36.96768728778939),
                    (-122.66944064253451, 34.10377172691159),
                    (-117.75040020737816, 34.10377172691159),
                    (-117.75040020737816, 36.96768728778939),
                    (-122.66944064253451, 36.96768728778939),
                ]
            )
        ],
        crs="epsg:4326",
    )
    with pytest.raises(exceptions.InvalidSize):
        roi.ROI(rois_gdf=rectangle)


# Test that when ROI is not a polygon an error is thrown
def test_ROI_wrong_geometry():
    line = gpd.GeoDataFrame(
        geometry=[
            LineString(
                [
                    (-120.83849150866949, 35.43786191889319),
                    (-120.93431712689429, 35.40749430666743),
                ]
            )
        ],
        crs="epsg:4326",
    )
    with pytest.raises(exceptions.InvalidGeometryType):
        roi.ROI(rois_gdf=line)


# create ROIs from geodataframe with a different CRS
def test_initialize_from_roi_gdf_different_crs():
    CRS = "epsg:2033"
    rectangle = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (-121.12083854611063, 35.56544740627308),
                    (-121.12083854611063, 35.53742390816822),
                    (-121.08749373817861, 35.53742390816822),
                    (-121.08749373817861, 35.56544740627308),
                    (-121.12083854611063, 35.56544740627308),
                ]
            )
        ],
        crs="epsg:4326",
    )
    rectangle.to_crs(CRS, inplace=True)
    rois = roi.ROI(rois_gdf=rectangle)
    assert hasattr(rois, "gdf")
    assert isinstance(rois.gdf, gpd.GeoDataFrame)
    assert rois.gdf.empty == False
    assert isinstance(rois.gdf.crs, pyproj.CRS)
    assert rois.gdf.crs == "epsg:4326"


# create ROIs from geodataframe with CRS 4326 (default CRS for map)
def test_initialize_from_roi_gdf():
    CRS = "epsg:4326"
    rectangle = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (-121.12083854611063, 35.56544740627308),
                    (-121.12083854611063, 35.53742390816822),
                    (-121.08749373817861, 35.53742390816822),
                    (-121.08749373817861, 35.56544740627308),
                    (-121.12083854611063, 35.56544740627308),
                ]
            )
        ],
        crs=CRS,
    )
    rois = roi.ROI(rois_gdf=rectangle)
    assert hasattr(rois, "gdf")
    assert isinstance(rois.gdf, gpd.GeoDataFrame)
    assert rois.gdf.empty == False
    assert isinstance(rois.gdf.crs, pyproj.CRS)
    assert rois.gdf.crs == CRS


def test_create_geodataframe(
    valid_bbox_gdf: gpd.GeoDataFrame,
    valid_shoreline_gdf: gpd.GeoDataFrame,
    valid_ROI: roi.ROI,
):
    large_len = 1000
    small_len = 750
    input_espg = "epsg:4326"

    actual_gdf = valid_ROI.create_geodataframe(
        bbox=valid_bbox_gdf,
        shoreline=valid_shoreline_gdf,
        large_length=large_len,
        small_length=small_len,
        crs=input_espg,
    )

    assert isinstance(actual_gdf, gpd.GeoDataFrame)
    assert set(actual_gdf.columns) == set(["geometry", "id"])
    assert actual_gdf.dtypes["geometry"] == "geometry"
    assert actual_gdf.dtypes["id"] == "object"
    # drop unneeded columns before checking
    columns_to_drop = list(valid_shoreline_gdf.columns.difference(["geometry"]))
    valid_shoreline_gdf = valid_shoreline_gdf.drop(columns=columns_to_drop)
    # Validate any shorelines intersect any squares in actual_gdf
    intersection_gdf = gpd.sjoin(
        valid_shoreline_gdf, right_df=actual_gdf, how="inner", predicate="intersects"
    )
    assert intersection_gdf.empty == False


def test_fishnet_intersection(
    valid_bbox_gdf: gpd.GeoDataFrame,
    valid_shoreline_gdf: gpd.GeoDataFrame,
    valid_ROI: roi.ROI,
):
    # tests if a valid fishnet geodataframe intersects with given shoreline geodataframe
    square_size = 1000
    output_epsg = "epsg:4326"
    fishnet_gdf = valid_ROI.create_rois(valid_bbox_gdf, square_size)
    # check if fishnet intersects the shoreline
    fishnet_gdf = valid_ROI.fishnet_intersection(fishnet_gdf, valid_shoreline_gdf)
    assert isinstance(fishnet_gdf, gpd.GeoDataFrame)
    assert fishnet_gdf.empty == False
    assert isinstance(fishnet_gdf.crs, pyproj.CRS)
    assert fishnet_gdf.crs == output_epsg
    assert set(fishnet_gdf.columns) == set(["geometry"])
    # drop unneeded columns before checking
    columns_to_drop = list(valid_shoreline_gdf.columns.difference(["geometry"]))
    valid_shoreline_gdf = valid_shoreline_gdf.drop(columns=columns_to_drop)
    # check if any shorelines intersect any squares in fishnet_gdf
    intersection_gdf = valid_shoreline_gdf.sjoin(
        fishnet_gdf, how="inner", predicate="intersects"
    )
    assert intersection_gdf.empty == False


def test_get_fishnet(
    valid_bbox_gdf: gpd.GeoDataFrame,
    valid_shoreline_gdf: gpd.GeoDataFrame,
    valid_ROI: roi.ROI,
):
    # tests if a valid fishnet geodataframe intersects with given shoreline geodataframe
    square_size = 1000
    output_epsg = "epsg:4326"
    # check if fishnet intersects the shoreline
    fishnet_gdf = valid_ROI.get_fishnet_gdf(
        bbox_gdf=valid_bbox_gdf,
        shoreline_gdf=valid_shoreline_gdf,
        square_length=square_size,
    )
    assert isinstance(fishnet_gdf, gpd.GeoDataFrame)
    assert isinstance(fishnet_gdf.crs, pyproj.CRS)
    assert fishnet_gdf.crs == output_epsg
    assert set(fishnet_gdf.columns) == set(["geometry"])
    # drop unneeded columns before checking
    columns_to_drop = list(valid_shoreline_gdf.columns.difference(["geometry"]))
    valid_shoreline_gdf = valid_shoreline_gdf.drop(columns=columns_to_drop)
    # check if any shorelines intersect any squares in fishnet_gdf
    intersection_gdf = valid_shoreline_gdf.sjoin(
        fishnet_gdf, how="inner", predicate="intersects"
    )
    assert intersection_gdf.empty == False


def test_roi_missing_lengths(valid_bbox_gdf, valid_shoreline_gdf):
    # test with missing square lengths
    with pytest.raises(Exception):
        roi.ROI(bbox=valid_bbox_gdf, shoreline=valid_shoreline_gdf)


def test_bad_roi_initialization(valid_bbox_gdf):
    empty_gdf = gpd.GeoDataFrame()
    # test with missing shoreline
    with pytest.raises(exceptions.Object_Not_Found):
        roi.ROI(bbox=valid_bbox_gdf)
    # test with missing bbox and shoreline
    with pytest.raises(exceptions.Object_Not_Found):
        roi.ROI()
    # test with empty bbox
    with pytest.raises(exceptions.Object_Not_Found):
        roi.ROI(bbox=empty_gdf)
    # test with empty shoreline
    with pytest.raises(exceptions.Object_Not_Found):
        roi.ROI(bbox=valid_bbox_gdf, shoreline=empty_gdf)


def test_roi_from_bbox_and_shorelines(valid_bbox_gdf, valid_shoreline_gdf):
    large_len = 1000
    small_len = 750
    actual_roi = roi.ROI(
        bbox=valid_bbox_gdf,
        shoreline=valid_shoreline_gdf,
        square_len_lg=large_len,
        square_len_sm=small_len,
    )

    assert isinstance(actual_roi, roi.ROI)
    assert isinstance(actual_roi.gdf, gpd.GeoDataFrame)
    assert set(actual_roi.gdf.columns) == set(["geometry", "id"])
    assert actual_roi.filename == "rois.geojson"
    assert hasattr(actual_roi, "extracted_shorelines")
    assert hasattr(actual_roi, "cross_shore_distances")
    assert hasattr(actual_roi, "roi_settings")


def test_create_fishnet(valid_bbox_gdf: gpd.GeoDataFrame, valid_ROI: roi.ROI):
    # tests if a valid geodataframe is created with square sizes approx. equal to given square_size
    square_size = 1000
    input_espg = "epsg:32610"
    output_epsg = "epsg:4326"

    # convert bbox to input_espg to most accurate espg to create fishnet with
    valid_bbox_gdf = valid_bbox_gdf.to_crs(input_espg)
    assert valid_bbox_gdf.crs == "epsg:32610"

    actual_fishnet = valid_ROI.create_fishnet(
        valid_bbox_gdf,
        input_espg=input_espg,
        output_epsg=output_epsg,
        square_size=square_size,
    )
    assert isinstance(actual_fishnet, gpd.GeoDataFrame)
    assert set(actual_fishnet.columns) == set(["geometry"])
    assert isinstance(actual_fishnet.crs, pyproj.CRS)
    assert actual_fishnet.crs == output_epsg
    # reproject back to input_espg to check if square sizes are correct
    actual_fishnet = actual_fishnet.to_crs(input_espg)
    # pick a square out of the fishnet ensure is approx. equal to square size
    actual_lengths = tuple(map(lambda x: x.length / 4, actual_fishnet["geometry"]))
    # check if actual lengths are close to square_size length
    is_actual_length_correct = all(
        tuple(
            map(lambda x: math.isclose(x, square_size, rel_tol=1e-04), actual_lengths)
        )
    )
    # assert all actual lengths are close to square_size length
    assert is_actual_length_correct == True


def test_create_rois(valid_ROI: roi.ROI, valid_bbox_gdf: gpd.GeoDataFrame):
    square_size = 1000
    # espg code of the valid_bbox_gdf
    input_espg = "epsg:4326"
    output_epsg = "epsg:4326"
    actual_roi_gdf = valid_ROI.create_rois(
        bbox=valid_bbox_gdf,
        input_espg=input_espg,
        output_epsg=output_epsg,
        square_size=square_size,
    )
    assert isinstance(actual_roi_gdf, gpd.GeoDataFrame)
    assert isinstance(actual_roi_gdf.crs, pyproj.CRS)
    assert actual_roi_gdf.crs == output_epsg
    assert set(actual_roi_gdf.columns) == set(["geometry"])


def test_transect_compatible_roi(transect_compatible_roi: gpd.GeoDataFrame):
    """tests if a ROI will be created from valid rois of type gpd.GeoDataFrame
    Args:
        valid_bbox_gdf (gpd.GeoDataFrame): alid rois as a gpd.GeoDataFrame
    """
    actual_roi = roi.ROI(rois_gdf=transect_compatible_roi)
    assert isinstance(actual_roi, roi.ROI)
    assert isinstance(actual_roi.gdf, gpd.GeoDataFrame)
    assert set(actual_roi.gdf.columns) == set(["geometry", "id"])
    assert actual_roi.gdf.dtypes["geometry"] == "geometry"
    assert actual_roi.gdf.dtypes["id"] == "object"
    assert actual_roi.filename == "rois.geojson"
    assert hasattr(actual_roi, "extracted_shorelines")
    assert hasattr(actual_roi, "cross_shore_distances")
    assert hasattr(actual_roi, "roi_settings")


def test_add_extracted_shoreline(valid_ROI: roi.ROI):
    """tests if a ROI will be created from valid rois of type gpd.GeoDataFrame
    Args:
       transect_compatible_roi (gpd.GeoDataFrame): valid rois as a gpd.GeoDataFrame
    """
    roi_id = "23"
    expected_dict = {
        "23": {
            "filename": ["ms.tif", "2019.tif"],
            "cloud_cover": [0.14, 0.0],
            "geoaccuracy": [7.9, 9.72],
            "idx": [4, 6],
            "MNDWI_threshold": [-0.231, -0.3],
            "satname": ["L8", "L8"],
        }
    }
    valid_ROI.add_extracted_shoreline(expected_dict[roi_id], roi_id)
    # valid_ROI.add_extracted_shoreline(expected_dict)
    assert valid_ROI.extracted_shorelines != {}
    # assert valid_ROI.extracted_shorelines == expected_dict
    assert valid_ROI.get_extracted_shoreline(roi_id) == expected_dict[roi_id]


def test_set_roi_settings(valid_ROI: roi.ROI):
    """tests if a ROI will be created from valid rois thats a gpd.GeoDataFrame
    Args:
        transect_compatible_roi (gpd.GeoDataFrame): valid rois as a gpd.GeoDataFrame
    """
    expected_dict = {
        23: {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID02022-10-07__09_hr_38_min37sec",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": 23,
            "polygon": [
                [
                    [-124.1662679688807, 40.863030239542944],
                    [-124.16690059058178, 40.89905645671534],
                    [-124.11942071317034, 40.89952713781644],
                    [-124.11881381876809, 40.863500326870245],
                    [-124.1662679688807, 40.863030239542944],
                ]
            ],
            "landsat_collection": "C01",
        },
        39: {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID12022-10-07__09_hr_38_min37sec",
            "filepath": "C:\\1_USGS\\CoastSeg\\repos\\2_CoastSeg\\CoastSeg_fork\\Seg2Map\\data",
            "roi_id": 39,
            "polygon": [
                [
                    [-124.16690059058178, 40.89905645671534],
                    [-124.1675343590045, 40.93508244001033],
                    [-124.12002870768146, 40.9355537155221],
                    [-124.11942071317034, 40.89952713781644],
                    [-124.16690059058178, 40.89905645671534],
                ]
            ],
            "landsat_collection": "C01",
        },
    }
    valid_ROI.set_roi_settings(expected_dict)
    assert valid_ROI.roi_settings != {}
    assert valid_ROI.roi_settings == expected_dict


def test_style_layer(valid_ROI: roi.ROI):
    with pytest.raises(AssertionError):
        valid_ROI.style_layer({}, layer_name="Nope")
# Test get_roi_settings method

def test_get_roi_settings_with_valid_roi_id(valid_ROI):
    # Test with valid ROI ID
    roi_id = "roi1"
    expected_settings = {"param1": 10, "param2": "abc"}
    valid_ROI.roi_settings = {roi_id: expected_settings}
    
    actual_settings = valid_ROI.get_roi_settings(roi_id)
    
    assert actual_settings == expected_settings

def test_get_roi_settings_with_invalid_roi_id(valid_ROI):
    # Test with invalid ROI ID
    roi_id = "roi2"
    valid_ROI.roi_settings = {"roi1": {"param1": 10, "param2": "abc"}}
    
    actual_settings = valid_ROI.get_roi_settings(roi_id)
    
    assert actual_settings == {}

def test_get_roi_settings_with_no_roi_id(valid_ROI):
    # Test with no ROI ID
    valid_ROI.roi_settings = {"roi1": {"param1": 10, "param2": "abc"}}
    
    actual_settings = valid_ROI.get_roi_settings()
    
    assert actual_settings == valid_ROI.roi_settings

def test_get_roi_settings_with_none_roi_id(valid_ROI):
    # Test with None ROI ID
    valid_ROI.roi_settings = {"roi1": {"param1": 10, "param2": "abc"}}
    
    with pytest.raises(ValueError):
        valid_ROI.get_roi_settings(None)

def test_get_roi_settings_with_non_string_roi_id(valid_ROI):
    # Test with non-string ROI ID
    roi_id = 123
    valid_ROI.roi_settings = {"roi1": {"param1": 10, "param2": "abc"}}
    
    with pytest.raises(TypeError):
        valid_ROI.get_roi_settings(roi_id)
        

def test_get_roi_settings_with_existing_roi_id(valid_ROI):
    roi_id = "roi1"
    valid_ROI.roi_settings = {roi_id: {"param1": 10, "param2": "abc"}}
    assert valid_ROI.get_roi_settings(roi_id) == {"param1": 10, "param2": "abc"}

def test_get_roi_settings_with_non_existing_roi_id(valid_ROI):
    roi_id = "roi1"
    valid_ROI.roi_settings = {"roi2": {"param1": 10, "param2": "abc"}}
    assert valid_ROI.get_roi_settings(roi_id) == {}

def test_get_roi_settings_with_empty_roi_id(valid_ROI):
    valid_ROI.roi_settings = {"roi1": {"param1": 10, "param2": "abc"}}
    assert valid_ROI.get_roi_settings("") == {"roi1": {"param1": 10, "param2": "abc"}}

def test_get_roi_settings_with_none_roi_id(valid_ROI):

    assert valid_ROI.get_roi_settings(None) == {}

def test_get_roi_settings_with_non_string_roi_id(valid_ROI):
    with pytest.raises(TypeError):
        valid_ROI.get_roi_settings(123)
        
        
def test_update_roi_settings_with_valid_settings(valid_ROI):
    new_settings = {
        "param1": 10,
        "param2": "value",
        "param3": [1, 2, 3]
    }
    valid_ROI.update_roi_settings(new_settings)
    assert valid_ROI.roi_settings == new_settings

def test_update_roi_settings_with_empty_settings(valid_ROI_with_settings):
    new_settings = {}
    valid_ROI_with_settings.update_roi_settings(new_settings)
    expected_settings = {"13": {
            "polygon": [
            [
                [-117.4684719510983, 33.265263693689256],
                [-117.46868751642162, 33.30560084719839],
                [-117.42064919876344, 33.30577275029851],
                [-117.42045572621824, 33.26543533468434],
                [-117.4684719510983, 33.265263693689256]
            ]
            ],
            "sitename": "ID_13_datetime06-05-23__04_16_45",
            "landsat_collection": "C02",
            "roi_id": "13",
            "sat_list": ["L8", "L9"],
            "filepath": r"C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
            "dates": ["2018-12-01", "2023-03-01"]
        },
        "12": {
            "polygon": [
            [
                [-117.4682568148693, 33.224926276845096],
                [-117.4684719510983, 33.265263693689256],
                [-117.42045572621824, 33.26543533468434],
                [-117.42026263879279, 33.22509765597134],
                [-117.4682568148693, 33.224926276845096]
            ]
            ],
            "sitename": "ID_12_datetime06-05-23__04_16_45",
            "landsat_collection": "C02",
            "roi_id": "12",
            "sat_list": ["L8", "L9"],
            "filepath": r"C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
            "dates": ["2018-12-01", "2023-03-01"]
        }
    }
    assert valid_ROI_with_settings.roi_settings.keys() == expected_settings.keys()
    assert valid_ROI_with_settings.roi_settings == expected_settings
    

def test_update_roi_settings_with_existing_settings(valid_ROI_with_settings):
    new_settings = {
        "13": {
        "polygon": [
        [
            [-117.4684719510983, 33.265263693689256],
            [-118.46868751642162, 33.30560084719839],
            [-117.42064919876344, 33.30577275029851],
            [-117.42045572621824, 33.26543533468434],
            [-117.4684719510983, 33.265263693689256]
        ]
        ],
        "sitename": "ID_13_datetime06-05-23__04_16_45",
        "landsat_collection": "C02",
        "roi_id": "13",
        "sat_list": ["L7", "L9"],
        "filepath": r"C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
        "dates": ["2017-12-01", "2023-03-01"]
    },
    }
    expected_settings = {        "13": {
        "polygon": [
        [
            [-117.4684719510983, 33.265263693689256],
            [-118.46868751642162, 33.30560084719839],
            [-117.42064919876344, 33.30577275029851],
            [-117.42045572621824, 33.26543533468434],
            [-117.4684719510983, 33.265263693689256]
        ]
        ],
        "sitename": "ID_13_datetime06-05-23__04_16_45",
        "landsat_collection": "C02",
        "roi_id": "13",
        "sat_list": ["L7", "L9"],
        "filepath": r"C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
        "dates": ["2017-12-01", "2023-03-01"]
        },
        "12": {
            "polygon": [
            [
                [-117.4682568148693, 33.224926276845096],
                [-117.4684719510983, 33.265263693689256],
                [-117.42045572621824, 33.26543533468434],
                [-117.42026263879279, 33.22509765597134],
                [-117.4682568148693, 33.224926276845096]
            ]
            ],
            "sitename": "ID_12_datetime06-05-23__04_16_45",
            "landsat_collection": "C02",
            "roi_id": "12",
            "sat_list": ["L8", "L9"],
            "filepath": r"C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
            "dates": ["2018-12-01", "2023-03-01"]
        },
    }
    valid_ROI_with_settings.update_roi_settings(new_settings)
    assert valid_ROI_with_settings.roi_settings == expected_settings

def test_update_roi_settings_with_none_settings(valid_ROI):
    with pytest.raises(ValueError):
        valid_ROI.update_roi_settings(None)