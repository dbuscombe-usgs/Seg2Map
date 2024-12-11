import os
import copy
import logging
from typing import Callable, Dict, List
from typing import Union

# Internal dependencies imports
from coastseg import shoreline
from coastseg import transects
from coastseg import shoreline_extraction_area
import geopandas as gpd
import pandas as pd
from typing import List, Callable, Any


logger = logging.getLogger(__name__)


def edit_geojson_files(
    filepaths: List[str],
    filter_function: Callable[[List[Any], gpd.GeoDataFrame], gpd.GeoDataFrame],
    **kwargs,
) -> None:
    """
    Applies a filter function to GeoDataFrame objects loaded from GeoJSON files,
    and writes the results back to the files.

    Each of the keywoard argument should be a column name in the GeoDataFrame with the values
    to filter in the given column. The filter function should take a GeoDataFrame and the kwargs.

    Args:
    filepaths (list): A list of strings representing the locations of the GeoJSON files to be edited.
    filter_function (Callable): A function that takes a list of selected items and a GeoDataFrame,
                            applies some filtering or editing based on the selected items,
                            and returns the modified GeoDataFrame.
    Returns:
     None
    """
    for filepath in filepaths:
        if os.path.exists(filepath):
            edit_gdf_file(filepath, filter_function, **kwargs)


def edit_gdf_file(
    filepath: str,
    filter_function: Callable[[gpd.GeoDataFrame, Dict[str, Any]], gpd.GeoDataFrame],
    **kwargs: Any,
) -> None:
    """
    Applies a filter function to a GeoDataFrame object loaded from a GeoJSON file,
    and writes the result back to the file.

    Each of the keywoard argument should be a column name in the GeoDataFrame with the values
    to filter in the given column. The filter function should take a GeoDataFrame and the kwargs.

    :param filepath: A string representing the path to the GeoJSON file.
    :param filter_function: A function that takes a GeoDataFrame and kwargs,
                            applies some filtering or editing based on the kwargs,
                            and returns the modified GeoDataFrame.
    :param kwargs: Additional keyword arguments to be passed to the filter function.
    :return: None
    """
    gdf = read_gpd_file(filepath)
    new_gdf = filter_function(gdf, **kwargs)
    new_gdf.to_file(filepath, driver="GeoJSON")


def create_geofeature_geodataframe(
    geofeature_path: str, roi_gdf: gpd.GeoDataFrame, epsg_code: str, feature_type: str
) -> gpd.GeoDataFrame:
    """
    Creates geofeature (shoreline or transects) as a GeoDataFrame.

    If the geofeature file exists at the provided path, it is read into a GeoDataFrame.
    If the GeoDataFrame is not empty, geofeature are extracted using the appropriate class
    (Shoreline or Transects) and a deep copy of the resulting GeoDataFrame is made.
    If the geofeature file does not exist, geofeatures are created based on the ROI GeoDataFrame.
    The resulting GeoDataFrame is then converted to the specified EPSG code.

    Args:
        geofeature_path (str): Path to the geofeature file.
        roi_gdf (gpd.GeoDataFrame): GeoDataFrame representing the region of interest.
        epsg_code (str): EPSG code for the desired coordinate reference system.
        feature_type (str): Type of geofeature (e.g., 'shoreline' or 'transect')

    Returns:
        gpd.GeoDataFrame: Geofeatures as a GeoDataFrame in the specified EPSG code.

    Raises:
        ValueError: If the geofeature file is empty.
    """
    feature_type = (
        feature_type.lower()
    )  # Convert to lower case for case insensitive comparison

    if os.path.exists(geofeature_path):
        # Load features from file
        geofeature_gdf = load_feature_from_file(geofeature_path, feature_type)
    else:
        # if a geofeature file is not given load features from ROI
        geofeature_gdf = load_geofeatures_from_roi(roi_gdf, feature_type)

    logger.info(f"{feature_type}_gdf: length {len(geofeature_gdf)} sample {geofeature_gdf.head(1)}")
    if geofeature_gdf.empty:
        raise Exception(
            f"None of the {feature_type}s intersected the ROI. Try a different {feature_type}"
        )
    geofeature_gdf = geofeature_gdf.loc[:, ["id", "geometry"]]
    geofeature_gdf = geofeature_gdf.to_crs(epsg_code)
    return geofeature_gdf


FEATURE_TYPE_MAP = {
    "transect": transects.Transects,
    "transects": transects.Transects,
    "shoreline": shoreline.Shoreline,
    "shorelines": shoreline.Shoreline,
    "reference_shoreline": shoreline.Shoreline,
    "reference_shorelines": shoreline.Shoreline,
    "shoreline_extraction_area": shoreline_extraction_area.Shoreline_Extraction_Area,
}

def load_geofeatures_from_roi(
    roi_gdf: gpd.GeoDataFrame, feature_type: str
) -> gpd.GeoDataFrame:
    """
    Given a Region Of Interest (ROI), this function attempts to load any geographic features (transects or shorelines)
    that exist in that region. If none exist, the user is advised to upload their own file.

    Args:
        roi_gdf (gpd.GeoDataFrame): GeoDataFrame representing the region of interest.
        feature_type (str): Type of the geographic feature, e.g. 'shoreline', 'transect'.

    Returns:
        gpd.GeoDataFrame: Geographic features as a GeoDataFrame.

    Raises:
        ValueError: If no geographic features were found in the given ROI.
    """
    feature_type = feature_type.lower()  # Convert to lower case for case insensitive comparison

    if feature_type in FEATURE_TYPE_MAP:
        feature_object = FEATURE_TYPE_MAP[feature_type](bbox=roi_gdf)
    else:
        logger.error(f"Unsupported feature_type: {feature_type}")
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    if feature_object.gdf.empty:
        logger.error(
            f"CoastSeg currently does not have {feature_type}s available in this region. Try uploading your own {feature_type}s.geojson"
        )
        raise Exception(
            f"CoastSeg currently does not have {feature_type}s available in this region. Try uploading your own {feature_type}s.geojson"
        )

    feature_gdf = copy.deepcopy(feature_object.gdf)

    return feature_gdf


def read_gpd_file(filename: str) -> gpd.GeoDataFrame:
    """
    Returns geodataframe from geopandas geodataframe file
    """
    if os.path.exists(filename):
        logger.info(f"Opening \n {filename}")
        return gpd.read_file(filename)
    else:
        raise FileNotFoundError(filename)


def load_geodataframe_from_file(
    feature_path: str, feature_type: str
) -> gpd.GeoDataFrame:
    """
    Load a geographic feature from a file. The file is read into a GeoDataFrame.
    Can read both geojson files and config_gdf.geojson files

    Args:
        feature_path (str): Path to the feature file.
        feature_type (str): Type of the geographic feature, e.g. 'shoreline', 'transect','rois','bbox'

    Returns:
        gpd.GeoDataFrame: Geographic feature as a GeoDataFrame.

    Raises:
        ValueError: If the feature file is empty.
    """
    logger.info(f"Attempting to load {feature_type} from a file")
    original_feature_gdf = read_gpd_file(feature_path)
    try:
        # attempt to load features from a config file
        feature_gdf = extract_feature_from_geodataframe(
            original_feature_gdf, feature_type=feature_type
        )
    except ValueError as e:
        # if it isn't a config file then just ignore the error
        logger.info(f"This probably wasn't a config : {feature_path} \n {e}")
        feature_gdf = original_feature_gdf # if this wasn't a config then just return the original file


    if feature_gdf.empty:
        raise ValueError(f"Empty {feature_type} file provided: {feature_path}")
    return feature_gdf


def load_feature_from_file(feature_path: str, feature_type: str):
    gdf = load_geodataframe_from_file(feature_path, feature_type)
    feature = create_feature(feature_type, gdf)
    feature_gdf = copy.deepcopy(feature.gdf)
    return feature_gdf


def create_feature(feature_type: str, gdf):
    feature_type = feature_type.lower()  # Convert to lower case for case insensitive comparison
    if feature_type in FEATURE_TYPE_MAP:
        if "transect" in feature_type:
            feature_object = FEATURE_TYPE_MAP[feature_type](transects=gdf)
        elif "shoreline" in feature_type:
            feature_object = FEATURE_TYPE_MAP[feature_type](shoreline=gdf)
        elif "roi" in feature_type:
            feature_object = FEATURE_TYPE_MAP[feature_type](rois_gdf=gdf)
        else:
            feature_object = FEATURE_TYPE_MAP[feature_type](gdf)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    return feature_object


def extract_feature_from_geodataframe(
    gdf: gpd.GeoDataFrame, feature_type: str, type_column: str = "type"
) -> gpd.GeoDataFrame:
    """
    Extracts a GeoDataFrame of features of a given type from a larger GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the features to extract.
        feature_type (str): The type of feature to extract. Typically one of the following 'shoreline', 'rois', 'transects', 'bbox'.
        type_column (str, optional): The name of the column containing feature types. Defaults to 'type'.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame containing only the features of the specified type.

    Raises:
        ValueError: Raised when feature_type or the type_column do not exist in the GeoDataFrame.
    """
    # Convert column names to lower case for case-insensitive matching
    gdf.columns = gdf.columns.str.lower()
    type_column = type_column.lower()

    # Check if type_column exists in the GeoDataFrame
    if type_column not in gdf.columns:
        raise ValueError(
            f"Column '{type_column}' does not exist in the GeoDataFrame. Incorrect config_gdf.geojson loaded"
        )

    # Handling pluralization of feature_type
    feature_types = {
        feature_type.lower(),
        (feature_type + "s").lower(),
        (feature_type.rstrip("s")).lower(),
    }

    # Filter the GeoDataFrame for the specified types
    filtered_gdf = gdf[gdf[type_column].str.lower().isin(feature_types)]

    return filtered_gdf
