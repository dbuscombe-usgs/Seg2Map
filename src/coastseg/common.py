# Standard library imports
import os
import re
import glob
import shutil
import json
import math
import logging
import random
import string
from datetime import datetime


# Third-party imports
import requests
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from area import area
from tqdm.auto import tqdm
from PIL import Image
from ipyfilechooser import FileChooser
from ipywidgets import ToggleButton, HBox, VBox, Layout, HTML
from requests.exceptions import SSLError
from shapely.geometry import Polygon

# Specific classes/functions from modules
from typing import Callable, List, Optional, Union, Dict, Set

# Internal dependencies imports
from coastseg import exceptions
from coastseg.validation import find_satellite_in_filename
from coastseg import file_utilities
from coastseg.exceptions import InvalidGeometryType

# widget icons from https://fontawesome.com/icons/angle-down?s=solid&f=classic

# Logger setup
logger = logging.getLogger(__name__)


def filter_images_by_roi(roi_settings: list[dict]):
    """
    Filters images in specified locations based on their Regions of Interest (ROI).

    This function iterates over the given list of ROI settings dictionaries. For each ROI,
    it constructs a GeoDataFrame and filters images located in a predefined directory based
    on the constructed ROI. The function logs a warning and skips to the next ROI
    if the specified directory for an ROI does not exist.

    Args:
        roi_settings (list[dict]): A list of dictionaries, each containing the settings
                                   for a Region of Interest (ROI). Each dictionary must
                                   have the following structure:
                                   {
                                     'roi_id': <int>,
                                     'sitename': <str>,
                                     'filepath': <str>,  # Base filepath for the ROI
                                     'polygon': <list>,  # List of coordinates representing the ROI polygon
                                   }

    Returns:
        None: This function doesn't return anything.

    Raises:
        KeyError: If a required key ('sitename', 'filepath', 'polygon') is missing
                  in any of the dictionaries in roi_settings.

    Logs:
        A warning if the location specified by 'filepath' and 'sitename', or the 'ROI_jpg_location'
        does not exist, specifying the nonexistent location.

    Example:
        >>> roi_settings = [
        ...     {
        ...         'roi_id': 1,
        ...         'sitename': 'site1',
        ...         'filepath': '/path/to/site1',
        ...         'polygon': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],
        ...     },
        ...     # More dictionaries for other ROIs
        ... ]
        >>> filter_images_by_roi(roi_settings)
    """
    # loop through each roi's settings by id
    for roi_id in roi_settings.keys():
        sitename = roi_settings[roi_id]["sitename"]
        filepath = roi_settings[roi_id]["filepath"]
        polygon = Polygon(roi_settings[roi_id]["polygon"][0])
        roi_location = os.path.join(filepath, sitename)
        if not os.path.exists(roi_location):
            logger.warning(f"Could not filter {roi_location} did not exist")
            continue
        ROI_jpg_location = os.path.join(
            roi_location, "jpg_files", "preprocessed", "RGB"
        )
        if not os.path.exists(ROI_jpg_location):
            logger.warning(f"Could not filter {ROI_jpg_location} did not exist")
            continue
        roi_gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs="EPSG:4326")
        bad_images = filter_partial_images(roi_gdf, ROI_jpg_location)
        logger.info(f"Partial images filtered out: {bad_images}")


def filter_partial_images(
    roi_gdf: gpd.GeoDataFrame,
    directory: str,
    min_area_percentage: float = 0.60,
    max_area_percentage: float = 1.5,
):
    """
    Filters images in a directory based on their area with respect to the area of the Region of Interest (ROI).

    This function uses the specified area percentages of the ROI to create a permissible area range. It then checks
    each image in the directory against this permissible range, and filters out any images whose areas are outside
    this range.

    Args:
        roi_gdf (GeoDataFrame): A GeoDataFrame containing the geometry of the ROI.
        directory (str): Directory path containing the images to be filtered.
        min_area_percentage (float, optional): Specifies the minimum area percentage of the ROI. For instance,
            0.60 indicates that the minimum permissible area is 60% of the ROI area. Defaults to 0.60.
        max_area_percentage (float, optional): Specifies the maximum area percentage of the ROI. For instance,
            1.5 indicates that the maximum permissible area is 150% of the ROI area. Defaults to 1.5.

    Returns:
        None: This function doesn't return any value but instead acts on the image files in the directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist or doesn't contain any image files.

    Example:
        >>> roi_gdf = geopandas.read_file('path_to_roi_file')
        >>> filter_partial_images(roi_gdf, 'path_to_image_directory')
    """
    # low and high range are in km
    roi_area = get_roi_area(roi_gdf)
    filter_images(
        roi_area * min_area_percentage, roi_area * max_area_percentage, directory
    )


def get_roi_area(gdf: gpd.GeoDataFrame) -> float:
    """
    Calculates the area of the Region of Interest (ROI) from the given GeoDataFrame.

    The function re-projects the GeoDataFrame to the appropriate UTM zone before calculating the area to ensure accurate area measurements.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame containing the geometry of the ROI. Assumes that the GeoDataFrame has at least one geometry.

    Returns:
        float: The area of the ROI in square kilometers.

    Raises:
        IndexError: If the GeoDataFrame is empty.
        ValueError: If the re-projection to the UTM zone fails.

    Example:
        >>> gdf = geopandas.read_file('path_to_file')
        >>> get_roi_area(gdf)
        12.34  # example output in km^2
    """
    # before  getting the most accurate epsg code convert it to CRS epsg 4326
    gdf = gdf.to_crs("epsg:4326")
    epsg_code = get_epsg_from_geometry(gdf.geometry.iloc[0])
    # re-project to the UTM zone
    projected_gdf = gdf.to_crs(epsg_code)
    # calculate the area in km^2
    return projected_gdf.area.iloc[0] / 1e6


def get_satellite_name(filename: str):
    """Returns the satellite name in the jpg name. Does not work tiffs"""
    try:
        return filename.split("_")[2].split(".")[0]
    except IndexError:
        # logger.error(f"Unable to extract satellite name from filename: {filename}")
        return None


def filter_images(
    min_area: float, max_area: float, directory: str, output_directory: str = ""
) -> list:
    """
    Filters images in a given directory based on a range of acceptable areas and moves the filtered out
    images to a specified output directory.

    The function calculates the area of each image in the specified directory. If the area is outside of
    the specified minimum and maximum area range, it's considered a bad image. The bad images are then
    moved to the output directory.

    Args:
        min_area (float): The minimum acceptable area in square kilometers.
        max_area (float): The maximum acceptable area in square kilometers.
        directory (str): The path to the directory containing the images to be filtered.
        output_directory (str, optional): The path to the directory where the bad images will be moved.
                                         If not provided, a new directory named 'bad' will be created
                                         inside the given directory.

    Returns:
        None: This function doesn't return anything; it moves the filtered out images to the specified
              output directory.

    Raises:
        FileNotFoundError: If the specified directory doesn't exist or doesn't contain any .jpg files.
        KeyError: If the satellite name extracted from a filename is not present in the predefined
                  pixel_size_per_satellite dictionary.

    Example:
        >>> filter_images(1, 10, 'path/to/images', 'path/to/bad_images')
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")

    if not output_directory:
        output_directory = os.path.join(directory, "bad")
    os.makedirs(output_directory, exist_ok=True)

    pixel_size_per_satellite = {
        "S2": 10,
        "L7": 15,
        "L8": 15,
        "L9": 15,
        "L5": 30,
    }
    bad_files = []
    jpg_files = [
        entry.name
        for entry in os.scandir(directory)
        if entry.is_file() and entry.name.lower().endswith(".jpg")
    ]

    for file in jpg_files:
        # Open the image and get dimensions
        satname = get_satellite_name(os.path.basename(file))
        if satname not in pixel_size_per_satellite:
            logger.error(
                f"Unknown satellite name {satname} extracted from filename: {file}"
            )
            continue

        filepath = os.path.join(directory, file)
        with Image.open(filepath) as img:
            width, height = img.size
            img_area = (
                width
                * pixel_size_per_satellite[satname]
                * height
                * pixel_size_per_satellite[satname]
            )
            img_area /= 1e6  # convert to square kilometers
            if img_area > max_area or img_area < min_area:
                bad_files.append(file)
    bad_files = list(map(lambda s: os.path.join(directory, s), bad_files))
    # move the bad files to the bad folder
    file_utilities.move_files(bad_files, output_directory)
    return bad_files  # Optionally return the list of bad files


def validate_geometry_types(
    gdf: gpd.GeoDataFrame,
    valid_types: set,
    feature_type: str = "Feature",
    help_message: str = None,
) -> None:
    """
    Check if all geometries in a GeoDataFrame are of the given valid types.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the geometries to check.
        valid_types (set): A set of valid geometry types.
        feature_type (str): The name of the feature

    Raises:
        ValueError: If any geometry in the GeoDataFrame is not of a type in valid_types.
    """

    # Extract the geometry types of the GeoDataFrame
    geometry_types = gdf.geometry.geom_type.unique()

    for geom_type in geometry_types:
        if geom_type not in valid_types:
            raise InvalidGeometryType(
                f"The {feature_type} contained a geometry of type '{geom_type}'",
                feature_name=feature_type,
                expected_geom_types=valid_types,
                wrong_geom_type=geom_type,
                help_msg=help_message,
            )
            # raise ValueError(
            #     f"The {feature_type} contained a geometry of type '{geom_type}' which is not in the list of valid types: {valid_types}"
            # )


def get_roi_polygon(
    roi_gdf: gpd.GeoDataFrame, roi_id: int
) -> Optional[List[List[float]]]:
    """
    Extract polygon coordinates for a given ROI ID from a GeoDataFrame.

    Parameters:
    - roi_gdf (gpd.GeoDataFrame): GeoDataFrame with "id" and "geometry" columns.
    - roi_id (int): ID of the region of interest.

    Returns:
    - Optional[List[List[float]]]: Polygon vertices or None if ROI ID is not found.

    Example:
    >>> polygon = get_roi_polygon(gdf, 1)
    """
    """Extract the polygonal geometry for a given ROI ID."""
    geoseries = roi_gdf[roi_gdf["id"] == roi_id]["geometry"]
    if not geoseries.empty:
        return [[[x, y] for x, y in list(geoseries.iloc[0].exterior.coords)]]
    return None


def get_cert_path_from_config(config_file="certifications.json"):
    """
    Get the certification path from the given configuration file.

    This function checks if the configuration file exists, reads the config file contents, and gets the certification path.
    If the certification path found in the config file is a valid file, it returns the certification path. Otherwise,
    it returns an empty string.

    Args:
        config_file (str): The path to the configuration file containing the certification path. Default is 'certifications.json'.

    Returns:
        str: The certification path if the config file exists and has a valid certification path, else an empty string.
    """
    logger.info(f"os.path.exists(config_file): {os.path.exists(config_file)}")
    if os.path.exists(config_file):
        # Read the config file
        with open(config_file, "r") as f:
            config_string = f.read()
            logger.info(f"certifications.json contents: {config_string}")
        try:
            config = json.loads(config_string)
        except json.JSONDecodeError:
            config_string = config_string.replace("\\", "\\\\")
            config = json.loads(config_string)

        # Get the cert path
        cert_path = config.get("cert_path")
        logger.info(f"certifications.json cert_path: {cert_path}")

        # If the cert path is a valid file, return it
        if cert_path and os.path.isfile(cert_path):
            logger.info(f"certifications.json cert_path isfile: {cert_path}")
            return cert_path

    # If the config file doesn't exist, or the cert path isn't in it, or the cert path isn't a valid file, return an empty string
    return ""


def get_response(url, stream=True):
    """
    Get the response from the given URL with or without a certification path.

    This function uses the get_cert_path_from_config() function to get a certification path, then sends an HTTP request (GET) to the
    specified URL. The certification is used if available, otherwise the request is sent without it. The stream parameter
    defines whether or not the response should be loaded progressively, and is set to True by default.

    Args:
        url (str): The URL to send the request to.
        stream (bool): If True, loads the response progressively (default True).

    Returns:
        requests.models.Response: The HTTP response object.
    """
    # attempt a standard request then try with an ssl certificate
    try:
        response = requests.get(url, stream=stream)
    except SSLError as e:
        cert_path = get_cert_path_from_config()
        if cert_path:  # if an ssl file was provided use it
            response = requests.get(url, stream=stream, verify=cert_path)
        else:  # if no ssl was provided
            raise exceptions.WarningException(
                "An SSL Verfication Error occured",
                "Save the location of your SSL certification file to certifications.json when downloading over a secure network",
            )
    return response


def filter_metadata(metadata: dict, sitename: str, filepath_data: str) -> dict[str]:
    """
    This function filters metadata to include only those files that exist in the given directory.

    Parameters:
    -----------
    metadata : dict
        The metadata dictionary to be filtered.

    sitename : str
        The site name used for filtering.

    filepath_data : str
        The base filepath where the data is located.

    Returns:
    --------
    dict
        The filtered metadata dictionary.
    """
    # Get the RGB directory
    RGB_directory = os.path.join(
        filepath_data, sitename, "jpg_files", "preprocessed", "RGB"
    )
    if not os.path.exists(RGB_directory):
        raise FileNotFoundError(
            f"Cannot extract shorelines from imagery. RGB directory did not exist. {RGB_directory}"
        )
    # filter out files that were removed from RGB directory
    filtered_files = get_filtered_files_dict(RGB_directory, "jpg", sitename)
    metadata = edit_metadata(metadata, filtered_files)
    return metadata


def edit_metadata(
    metadata: Dict[str, Dict[str, Union[str, List[Union[str, datetime, int, float]]]]],
    filtered_files: Dict[str, Set[str]],
) -> Dict[str, Dict[str, Union[str, List[Union[str, datetime, int, float]]]]]:
    """Filters the metadata so that it contains the data for the filenames in filered_files

    Args:
        metadata (dict): A dictionary containing the metadata for each satellite
        Each satellite has the following key fields "filenames","epsg","dates","acc_georef"
        Example:
        metadata = {
            'L8':{
                "filenames": ["2019-02-16-18-22-17_L8_sitename_ms.tif","2012-02-16-18-22-17_L8_sitename_ms.tif"],
                "epsg":[4326,4326],
                "dates":[datetime.datetime(2022, 1, 26, 15, 33, 50, tzinfo=<UTC>),datetime.datetime(2012, 1, 26, 15, 33, 50, tzinfo=<UTC>)],
                "acc_georef":[9.185,9.125],
            }
            'L9':{
                "filenames": ["2019-02-16-18-22-17_L9_sitename_ms.tif"],
                "epsg":[4326],
                "dates":[datetime.datetime(2022, 1, 26, 15, 33, 50, tzinfo=<UTC>)],
                "acc_georef":[9.185],
            }
        }
        filtered_files (dict): A dictionary containing a set of the tif filenames available for each satellite
        Example:
        filtered_files = {
            "L5": {},
            "L7": {},
            "L8": {"2019-02-16-18-22-17_L8_sitename_ms.tif"},
            "L9": {"2019-02-16-18-22-17_L9_sitename_ms.tif"},
            "S2": {},
        }

    Returns:
        dict: a filtered dictionary containing only the data for the filenames in filtered_files
        Example:
                metadata = {
            'L8':{
                "filenames": ["2019-02-16-18-22-17_L8_sitename_ms.tif"],
                "epsg":[4326],
                "dates":[datetime.datetime(2022, 1, 26, 15, 33, 50, tzinfo=<UTC>)],
                "acc_georef":[9.185],
            }
            'L9':{
                "filenames": ["2019-02-16-18-22-17_L9_sitename_ms.tif"],
                "epsg":[4326],
                "dates":[datetime.datetime(2022, 1, 26, 15, 33, 50, tzinfo=<UTC>)],
                "acc_georef":[9.185],
            }
        }
    """
    # Iterate over satellite names in filtered_files
    for sat_name, files in filtered_files.items():
        # Check if sat_name is present in metadata
        if sat_name in metadata:
            satellite_metadata = metadata[sat_name]

            # Find the indices to keep based on filenames in filtered_files
            indices_to_keep = [
                idx
                for idx, filename in enumerate(satellite_metadata["filenames"])
                if filename in files
            ]

            # Loop through each key in the satellite_metadata dictionary
            for key, values in satellite_metadata.items():
                # Check if values is a list
                if isinstance(values, list):
                    if indices_to_keep:
                        # If indices_to_keep is not empty, filter the list based on it
                        satellite_metadata[key] = [values[i] for i in indices_to_keep]
                    else:
                        # If indices_to_keep is empty, assign an empty list
                        satellite_metadata[key] = []
    return metadata


def get_filtered_files_dict(directory: str, file_type: str, sitename: str) -> dict:
    """
    Scans the directory for files of a given type and groups them by satellite names into a dictionary.
    Each entry in the dictionary contains a set of multispectral tif filenames associated with the original filenames and site name.

    Example :
    file_type = "tif"
    sitename = "ID_onn15_datetime06-07-23__01_02_19"
    {
        "L5":{2014-12-19-18-22-40_L5_ID_onn15_datetime06-07-23__01_02_19_ms.tif,},
        "L7":{},
        "L8":{2014-12-19-18-22-40_L8_ID_onn15_datetime06-07-23__01_02_19_ms.tif,},
        "L9":{},
        "S2":{},
    }

    Parameters:
    -----------
    directory : str
        The directory where the files are located.

    file_type : str
        The filetype of the files to be included.
        Ex. 'jpg'

    sitename : str
        The site name to be included in the new filename.

    Returns:
    --------
    dict
        a dictionary where each key is a satellite name and each value is a set of the tif filenames.
    """
    filepaths = glob.iglob(os.path.join(directory, f"*.{file_type}"))

    satellites = {"L5": set(), "L7": set(), "L8": set(), "L9": set(), "S2": set()}
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        parts = filename.split("_")

        if len(parts) < 2:
            logging.warning(f"Skipping file with unexpected name format: {filename}")
            continue

        date = parts[0]

        satname = find_satellite_in_filename(filename)
        if satname is None:
            logging.warning(
                f"Skipping file with unexpected name format which was missing a satname: {filename}"
            )
            continue

        # satname_parts = parts[-1].split(".")

        # if len(satname_parts) < 2:
        #     logging.warning(
        #         f"Skipping file with unexpected name format: {old_filename}"
        #     )
        #     continue

        # satname = satname_parts[0]

        tif_filename = f"{date}_{satname}_{sitename}_ms.tif"
        if satname in satellites:
            satellites[satname].add(tif_filename)

    return satellites


def create_unique_ids(data, prefix_length: int = 3):
    # if not all the ids in data are unique
    if not check_unique_ids(data):
        # generate unique IDs with a matching prefix with the given length
        ids = generate_ids(num_ids=len(data), prefix_length=prefix_length)
        data["id"] = ids
    return data


def extract_feature_from_geodataframe(
    gdf: gpd.GeoDataFrame, feature_type: str, type_column: str = "type"
) -> gpd.GeoDataFrame:
    """
    Extracts a GeoDataFrame of features of a given type and specified columns from a larger GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the features to extract.
        feature_type (str): The type of feature to extract. Typically one of the following 'shoreline','rois','transects','bbox'
        type_column (str, optional): The name of the column containing feature types. Defaults to 'type'.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame containing only the features of the specified type and columns.

    Raises:
        ValueError: Raised when feature_type or any of the columns specified do not exist in the GeoDataFrame.
    """
    # Check if type_column exists in the GeoDataFrame
    if type_column not in gdf.columns:
        raise ValueError(
            f"Column '{type_column}' does not exist in the GeoDataFrame. Incorrect config_gdf.geojson loaded"
        )

    # select only the features that are of the correct type and have the correct columns
    feature_gdf = gdf[gdf[type_column] == feature_type]

    return feature_gdf


def random_prefix(length):
    """Generate a random string of the given length."""
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def generate_ids(num_ids, prefix_length):
    """Generate a list of sequential IDs with a random prefix.

    Args:
        num_ids (int): The number of IDs to generate.
        prefix_length (int): The length of the random prefix for the IDs.

    Returns:
        list: A list of IDs.
    """
    prefix = random_prefix(prefix_length)
    return [prefix + str(i) for i in range(1, num_ids + 1)]


def save_transects(
    roi_id: str,
    save_location: str,
    cross_distance_transects: dict,
    extracted_shorelines: dict,
    settings: dict,
) -> None:
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        roi_id (str): The ID of the ROI.
        save_location (str): The directory path to save the transect data.
        cross_distance_transects (dict): Dictionary containing cross distance transects data.
        extracted_shorelines (dict): Dictionary containing extracted shorelines data.

    Returns:
        None.
    """
    create_csv_per_transect(
        roi_id,
        save_location,
        cross_distance_transects,
        extracted_shorelines,
        filename="_timeseries_raw.csv",
    )
    save_transect_intersections(
        save_location,
        extracted_shorelines,
        cross_distance_transects,
        filename="transect_time_series.csv",
    )
    save_path = os.path.join(save_location, "transects_cross_distances.json")
    # save transect settings to file
    transect_settings = get_transect_settings(settings)
    transect_settings_path = os.path.join(save_location, "transects_settings.json")
    file_utilities.to_file(transect_settings, transect_settings_path)
    file_utilities.to_file(cross_distance_transects, save_path)


def get_downloaded_models_dir() -> str:
    """returns full path to downloaded_models directory and
    if downloaded_models directory does not exist then it is created
    Returns:
        str: full path to downloaded_models directory
    """
    # directory to hold downloaded models from Zenodo
    script_dir = os.path.dirname(os.path.abspath(__file__))

    downloaded_models_path = os.path.abspath(
        os.path.join(script_dir, "downloaded_models")
    )
    if not os.path.exists(downloaded_models_path):
        os.mkdir(downloaded_models_path)
    logger.info(f"downloaded_models_path: {downloaded_models_path}")

    return downloaded_models_path


def get_value_by_key_pattern(d: dict, patterns: list | set | tuple):
    """
    Function to extract the value from the first key in a dictionary that matches a pattern.

    Parameters:
    d (dict): The dictionary from which to extract the value.
    patterns (list | set | tuple): Iterable of patterns to match keys in the dictionary against.
    The function returns the value of the first key that matches a pattern.

    Returns:
    The value from the dictionary corresponding to the first key that matches a pattern in patterns,
    or None if no matching keys are found.
    """
    for key in d:
        for pattern in patterns:
            if re.search(pattern, key, re.IGNORECASE):
                return d[key]
    raise KeyError(f"None of {patterns} matched keys in {d.keys()}")


def copy_configs(src: str, dst: str) -> None:
    """Copy config files from source directory to destination directory.

    Looks for files with names starting with "config_gdf" and ending with ".geojson"
    and a file named "config.json" in the source directory.

    Args:
        src (str): the source directory
        dst (str): the destination directory
    """
    # Get the list of files in the source directory
    files = os.listdir(src)
    # Loop through the files and copy the ones we need
    for file in files:
        if file.startswith("config_gdf") and file.endswith(".geojson"):
            config_gdf_path = os.path.join(src, file)
            dst_file = os.path.join(dst, "config_gdf.geojson")
            logger.info(f"Copying {config_gdf_path} to {dst_file}")
            shutil.copy(config_gdf_path, dst_file)
        elif file == "config.json":
            config_json_path = os.path.join(src, file)
            dst_file = os.path.join(dst, "config.json")
            logger.info(f"Copying {config_json_path} to {dst_file}")
            shutil.copy(config_json_path, dst_file)


def create_file_chooser(
    callback: Callable[[FileChooser], None],
    title: str = None,
    filter_pattern: str = None,
    starting_directory: str = None,
):
    """
    This function creates a file chooser and a button to close the file chooser.
    It takes a callback function and an optional title as arguments.
    It only searches for .geojson files.

    Args:
        callback (Callable[[FileChooser],None]): A callback function that which is called
        when a file is selected.
        title (str): Optional title for the file chooser.

    Returns:
        chooser (HBox): A HBox containing the file chooser and close button.
    """
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    # creates a unique instance of filechooser and button to close filechooser
    inital_path = os.getcwd()
    if starting_directory:
        inital_path = os.path.join(inital_path, starting_directory)
    geojson_chooser = FileChooser(inital_path)

    geojson_chooser.dir_icon = os.sep

    geojson_chooser.filter_pattern = ["*.geojson"]
    if filter_pattern:
        geojson_chooser.filter_pattern = [filter_pattern]

    geojson_chooser.title = "<b>Select a geojson file</b>"
    if title is not None:
        geojson_chooser.title = f"<b>{title}</b>"
    # callback function is called when a file is selected
    geojson_chooser.register_callback(callback)

    close_button = ToggleButton(
        value=False,
        tooltip="Close File Chooser",
        icon="times",
        button_style="primary",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    def close_click(change: dict):
        if change["new"]:
            geojson_chooser.close()
            close_button.close()

    close_button.observe(close_click, "value")
    chooser = HBox([geojson_chooser, close_button], layout=Layout(width="100%"))
    return chooser


def get_most_accurate_epsg(epsg_code: int, bbox: gpd.GeoDataFrame):
    """Returns most accurate epsg code based on lat and lon if output epsg
    was 4326 or 4327
    Args:
        epsg_code(int or str): current epsg code
        bbox (gpd.GeoDataFrame): geodataframe for bounding box on map
    Returns:
        int: epsg code that is most accurate or unchanged if crs not 4326 or 4327
    """
    if isinstance(epsg_code, str) and epsg_code.startswith("epsg:"):
        epsg_code = epsg_code.split(":")[1]
    epsg_code = int(epsg_code)
    # coastsat cannot use 4326 to extract shorelines so modify epsg_code
    if epsg_code == 4326 or epsg_code == 4327:
        geometry = bbox.iloc[0]["geometry"]
        epsg_code = get_epsg_from_geometry(geometry)
    return epsg_code


def create_dir_chooser(callback, title: str = None, starting_directory: str = "data"):
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    inital_path = os.path.join(os.getcwd(), starting_directory)
    if not os.path.exists(inital_path):
        inital_path = os.getcwd()
    # creates a unique instance of filechooser and button to close filechooser
    dir_chooser = FileChooser(inital_path)
    dir_chooser.dir_icon = os.sep
    # Switch to folder-only mode
    dir_chooser.show_only_dirs = True
    if title is not None:
        dir_chooser.title = f"<b>{title}</b>"
    dir_chooser.register_callback(callback)

    close_button = ToggleButton(
        value=False,
        tooltip="Close Directory Chooser",
        icon="times",
        button_style="primary",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    def close_click(change):
        if change["new"]:
            dir_chooser.close()
            close_button.close()

    close_button.observe(close_click, "value")
    chooser = HBox([dir_chooser, close_button])
    return chooser


def get_transect_settings(settings: dict) -> dict:
    transect_settings = {}
    transect_settings["max_std"] = settings.get("max_std")
    transect_settings["min_points"] = settings.get("min_points")
    transect_settings["along_dist"] = settings.get("along_dist")
    transect_settings["max_range"] = settings.get("max_range")
    transect_settings["min_chainage"] = settings.get("min_chainage")
    transect_settings["multiple_inter"] = settings.get("multiple_inter")
    transect_settings["prc_multiple"] = settings.get("prc_multiple")
    return transect_settings


def create_directory_in_google_drive(path: str, name: str) -> str:
    """
    Creates a new directory with the provided name in the given path.
    Raises FileNotFoundError if the given path does not exist.

    Parameters:
    path (str): path to the directory where the new directory will be created
    name (str): name of the new directory

    Returns:
    new_path (str): path to the newly created directory
    """
    new_path = os.path.join(path, name)
    if os.path.exists(path):
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
    else:
        raise FileNotFoundError(new_path)
    return new_path


def is_in_google_colab() -> bool:
    """
    Returns True if the code is running in Google Colab, False otherwise.
    """
    if os.getenv("COLAB_RELEASE_TAG"):
        return True
    else:
        return False


def get_ids_with_invalid_area(
    geometry: gpd.GeoDataFrame, max_area: float = 98000000, min_area: float = 0
) -> set:
    """
    Get the indices of geometries with areas outside the specified range.

    This function checks the areas of each geometry in a given GeoDataFrame. If the area
    is either greater than `max_area` or less than `min_area`, the index of that geometry
    is added to the set of invalid geometries.

    Note:
        - The provided GeoDataFrame is assumed to be in CRS EPSG:4326.
        - Returned areas are in meters squared.

    Args:
        geometry (gpd.GeoDataFrame): The GeoDataFrame containing the geometries to check.
        max_area (float, optional): The maximum allowable area for a valid geometry. Defaults to 98000000.
        min_area (float, optional): The minimum allowable area for a valid geometry. Defaults to 0.

    Returns:
        set: A set of indices corresponding to the geometries with areas outside the specified range.

    Raises:
        TypeError: If the provided geometry is not a GeoDataFrame.
    """
    if isinstance(geometry, gpd.GeoDataFrame):
        geometry = json.loads(geometry.to_json())
    if isinstance(geometry, dict):
        if "features" in geometry.keys():
            rows_drop = set()
            for i, feature in enumerate(geometry["features"]):
                roi_area = get_area(feature["geometry"])
                if roi_area >= max_area or roi_area <= min_area:
                    rows_drop.add(i)
            return rows_drop
    else:
        raise TypeError("Must be geodataframe")


def load_cross_distances_from_file(dir_path):
    transect_dict = None
    glob_str = os.path.join(dir_path, "*transects_cross_distances.json*")
    for file in glob.glob(glob_str):
        if os.path.basename(file) == "transects_cross_distances.json":
            transect_dict = file_utilities.load_data_from_json(file)

    if transect_dict is None:
        logger.warning(
            f"No transect cross shore distances could be loaded from {dir_path}"
        )
        return None

    # convert lists to np.array for each transect
    for key in transect_dict.keys():
        tmp = np.array(transect_dict[key])
        transect_dict[key] = tmp
    logger.info(f"Loaded transect cross shore distances from: {dir_path}")
    return transect_dict


def mount_google_drive(name: str = "CoastSeg") -> None:
    """
    If the user is running in Google Colab, the Google Drive will be mounted to the root directory
    "/content/drive/MyDrive" and a new directory will be created with the provided name.

    Parameters:
    name (str): The name of the directory to be created. Default is 'CoastSeg'.

    Returns:
    None
    """
    if is_in_google_colab():
        from google.colab import drive

        # default location google drive is mounted to
        root_dir = "/content/drive/MyDrive"
        # mount google drive to default home directory
        drive.mount("/content/drive", force_remount=True)
        # create directory with provided name in google drive
        new_path = create_directory_in_google_drive(root_dir, name)
        # change working directory to directory with name
        os.chdir(new_path)
    else:
        print("Not running in Google Colab.")


def create_hover_box(title: str, feature_html: HTML = HTML("")) -> VBox:
    """
    Creates a box with a title and optional HTML containing information about the feature that was
    last hovered over.
    The hover box has two buttons, an 'uncollapse' and 'collapse' button.
    The 'uncollapse' button opens the hover box to reveal details about the feature that was
    last hovered over, whereas the 'collapse' button hides the feature_html and just shows the default messages of
    'Hover over a feature' or 'Hover Data Available'.

    Parameters:
    title (str): The title of the hover box
    feature_html (HTML, optional): HTML of the feature to be displayed in the hover box

    Returns:
    container (VBox): Box with the given title and details about the feature given by feature_html
    """
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    # create title
    title = HTML(f"  <h4>{title} Hover  </h4>")
    # Default message shown when nothing has been hovered
    msg = HTML(f"Hover over a feature</br>")
    # open button allows user to see hover data
    uncollapse_button = ToggleButton(
        value=False,
        tooltip="Show hover data",
        icon="angle-down",
        button_style="info",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    # collapse_button collapses hover data
    collapse_button = ToggleButton(
        value=False,
        tooltip="Show hover data",
        icon="angle-up",
        button_style="info",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    # message tells user that data is available on hover
    container_content = VBox([msg])
    if feature_html.value == "":
        container_content.children = [msg]
    elif feature_html.value != "":
        container_content.children = [feature_html]

    # default configuration for container is in collapsed mode
    container_header = HBox([title, uncollapse_button])
    container = VBox([container_header])

    def uncollapse_click(change: dict):
        logger.info(change)
        if feature_html.value == "":
            container_content.children = [msg]
        elif feature_html.value != "":
            container_content.children = [feature_html]
        container_header.children = [title, collapse_button]
        container.children = [container_header, container_content]

    def collapse_click(change: dict):
        logger.info(change)
        container_header.children = [title, uncollapse_button]
        container.children = [container_header]

    collapse_button.observe(collapse_click, "value")
    uncollapse_button.observe(uncollapse_click, "value")
    return container


def create_warning_box(
    title: str = None,
    msg: str = None,
    instructions: str = None,
    msg_width: str = "75%",
    box_width: str = "50%",
) -> HBox:
    """
    Creates a warning box with a title and message that can be closed with a close button.

    Parameters:
    title (str, optional): The title of the warning box. Default is 'Warning'.
    msg (str, optional): The message of the warning box. Default is 'Something went wrong...'.
    msg_width (str, optional): The width of the warning message. Default is '75%'.
    box_width (str, optional): The width of the warning box. Default is '50%'.

    Returns:
        HBox: The warning box containing the title, message, and close button.
    """
    # create title
    if title is None:
        title = "Warning"
    warning_title = HTML(f"<h2 style='text-align: center;'>⚠️{title}</h2>")
    # create msg
    if msg is None:
        msg = "Something went wrong..."
    if instructions is None:
        instructions = ""
    warning_msg = HTML(
        f"<div style='max-height: 250px; overflow-x: hidden; overflow-y: visible; text-align: center;'>"
        f"<span style='color: red'>⚠️</span>{msg}"
        f"</div>"
    )
    instructions_msg = HTML(
        f"<div style='max-height: 250px; overflow-x: hidden; overflow-y: visible; text-align: center;'>"
        f"<span style='color: red'></span>{instructions}"
        f"</div>"
    )

    x_button = ToggleButton(
        value=False,
        tooltip="Close Warning Box",
        icon="times",
        button_style="danger",
        layout=Layout(height="28px", width="28px"),
    )

    close_button = ToggleButton(
        value=False,
        description="Close",
        tooltip="Close Warning Box",
        button_style="danger",
        layout=Layout(height="28px", width="60px"),
    )

    # create vertical box to hold title and msg
    warning_content = VBox(
        [warning_title, warning_msg, instructions_msg, close_button],
        layout=Layout(width=msg_width, max_width="95%"),
    )

    def close_click(change):
        if change["new"]:
            warning_content.close()
            x_button.close()
            close_button.close()
            warning_box.close()

    close_button.observe(close_click, "value")
    x_button.observe(close_click, "value")
    warning_box = HBox(
        [warning_content, x_button],
        layout=Layout(width=box_width, border="2px solid red"),
    )
    return warning_box


def clear_row(row: HBox):
    """close widgets in row/column and clear all children
    Args:
        row (HBox)(VBox): row or column
    """
    for index in range(len(row.children)):
        row.children[index].close()
    row.children = []


def download_url(url: str, save_path: str, filename: str = None, chunk_size: int = 128):
    """Downloads the data from the given url to the save_path location.
    Args:
        url (str): url to data to download
        save_path (str): directory to save data
        chunk_size (int, optional):  Defaults to 128.
    """
    logger.info(f"download url: {url}")
    # get a response from the url
    response = get_response(url, stream=True)
    with response as r:
        logger.info(r)
        if r.status_code == 404:
            logger.error(f"Error {r.status_code}. DownloadError: {save_path}")
            raise exceptions.DownloadError(os.path.basename(save_path))
        if r.status_code == 429:
            logger.error(f"Error {r.status_code}.DownloadError: {save_path}")
            raise Exception(
                "Zenodo has denied the request. You may have requested too many files at once."
            )
        if r.status_code != 200:
            logger.error(f"Error {r.status_code}. DownloadError: {save_path}")
            raise exceptions.DownloadError(os.path.basename(save_path))
        # check header to get content length, in bytes
        content_length = r.headers.get("Content-Length")
        if content_length:
            total_length = int(content_length)
            with open(save_path, "wb") as fd:
                with tqdm(
                    total=total_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {filename}",
                    initial=0,
                    ascii=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))
        else:
            logger.warning("Content length not found in response headers")


def get_center_point(coords: list) -> tuple:
    """returns the center point of rectangle specified by points coords
    Args:
        coords list[tuple(float,float)]: lat,lon coordinates
    Returns:
        tuple[float]: (center x coordinate, center y coordinate)
    """
    x1, y1 = coords[0][0], coords[0][1]
    x2, y2 = coords[2][0], coords[2][1]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    return center_x, center_y


def get_epsg_from_geometry(geometry: "shapely.geometry.polygon.Polygon") -> int:
    """Uses geometry of shapely rectangle in crs 4326 to return the most accurate
    utm code as a string of format 'epsg:utm_code'
    example: 'espg:32610'

    Args:
        geometry (shapely.geometry.polygon.Polygon): geometry of a rectangle

    Returns:
        int: most accurate epsg code based on lat lon coordinates of given geometry
    """
    rect_coords = geometry.exterior.coords
    center_x, center_y = get_center_point(rect_coords)
    utm_code = convert_wgs_to_utm(center_x, center_y)
    return int(utm_code)


def convert_wgs_to_utm(lon: float, lat: float) -> str:
    """return most accurate utm epsg-code based on lat and lng
    convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
    Args:
        lon (float): longitude
        lat (float): latitude
    Returns:
        str: new espg code
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band  # North
        return epsg_code
    epsg_code = "327" + utm_band  # South
    return epsg_code


def extract_roi_by_id(gdf: gpd.GeoDataFrame, roi_id: int) -> gpd.GeoDataFrame:
    """Returns geodataframe with a single ROI whose id matches roi_id.
       If roi_id is None returns gdf

    Args:
        gdf (gpd.GeoDataFrame): ROI geodataframe to extract ROI with roi_id from
        roi_id (int): id of the ROI to extract
    Raises:
        exceptions.Id_Not_Found: if id doesn't exist in ROI's geodataframe or self.rois.gdf is empty
    Returns:
        gpd.GeoDataFrame: ROI with id matching roi_id
    """
    if roi_id is None:
        single_roi = gdf
    else:
        # Select a single roi by id
        single_roi = gdf[gdf["id"].astype(str) == str(roi_id)]
        # if the id was not found in the geodataframe raise an exception
    if single_roi.empty:
        logger.error(f"Id: {id} was not found in {gdf}")
        raise exceptions.Id_Not_Found(id)
    logger.info(f"single_roi: {single_roi}")
    return single_roi


def get_area(polygon: dict) -> float:
    "Calculates the area of the geojson polygon using the same method as geojson.io"
    return round(area(polygon), 3)


def extract_roi_data(json_data: dict, roi_id: str, fields_of_interest: list = []):
    """
    Extracts the specified fields for a specific ROI from a JSON data dictionary.

    Args:
        json_data (dict): The JSON data dictionary.
        roi_id (str): The ID of the ROI to extract data for.

    Returns:
        dict: A dictionary containing the extracted fields for the ROI.

    Raises:
        ValueError: If the config.json file is invalid or the ROI ID is not found.

    """
    roi_data = extract_fields(json_data, roi_id, fields_of_interest)
    if not roi_data:
        raise ValueError(
            "Invalid config.json file detected. Please add the correct roi ids to the config.json file's 'roi_ids' and try again."
        )
    return roi_data


def extract_fields(data, key=None, fields_of_interest=None):
    """
    Extracts specified fields from a given dictionary.

    Args:
        data (dict): A dictionary containing the data to extract fields from.
        key (str, optional): A string representing the key to extract fields from in the dictionary.
        fields_of_interest (list[str], optional): A list of strings representing the fields to extract from the dictionary.
            If not provided, the default fields of interest will be used.

    Returns:
        dict: A dictionary containing the extracted fields.

    """
    extracted_data = {}
    fields_of_interest = fields_of_interest or {
        "dates",
        "sitename",
        "polygon",
        "roi_id",
        "sat_list",
        "landsat_collection",
        "filepath",
    }

    if key and key in data:
        for field in fields_of_interest:
            if field in data[key]:
                extracted_data[field] = data[key][field]
    else:
        for field in fields_of_interest:
            if field in data:
                extracted_data[field] = data[field]

    return extracted_data


def check_unique_ids(data: gpd.GeoDataFrame) -> bool:
    """
    Checks if all the ids in the 'id' column of a geodataframe are unique. If the 'id' column does not exist returns False

    Args:
        data (gpd.GeoDataFrame): A GeoDataFrame with an 'id' column.

    Returns:
        bool: True if all ids are unique, False otherwise.
    """
    if "id" not in data.columns:
        return False
    return not any(data["id"].duplicated())


def preprocess_geodataframe(
    data: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    columns_to_keep: List[str] = None,
    create_ids: bool = True,
    output_crs: str = None,
) -> gpd.GeoDataFrame:
    """
    This function preprocesses a GeoDataFrame. It performs several transformations:

    - If 'ID' column exists, it's renamed to lowercase 'id'.
    - Z-axis coordinates are removed from data.
    - If an 'id' column does not exist, it creates one with unique IDs generated by a function generate_ids()
      with prefix of length 3. This option can be turned off by setting the parameter create_ids=False.
    - If the list of columns_to_keep is provided, only those columns are retained in the data.

    Args:
        data (gpd.GeoDataFrame, optional): The input GeoDataFrame to be preprocessed.
            Defaults to an empty GeoDataFrame.
        columns_to_keep (List[str], optional): The list of column names to retain in the preprocessed DataFrame.
            Defaults to None, in which case all columns are kept.
        create_ids (bool, optional): Flag to decide whether to create 'id' column if it doesn't exist.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame: The preprocessed GeoDataFrame.
    """
    if not data.empty:
        # rename 'ID' to lowercase if it exists
        data.rename(columns={"ID": "id"}, inplace=True)

        # remove z-axis from data
        data = remove_z_coordinates(data)

        # if an 'id' column does not exist, create one with row indices as ids
        if create_ids:
            if "id" not in data.columns.str.lower():
                ids = generate_ids(num_ids=len(data), prefix_length=3)
                data["id"] = ids

        # if columns_to_keep is specified, keep only those columns
        if columns_to_keep:
            columns_to_keep = set(col.lower() for col in columns_to_keep)
            data = data[[col for col in data.columns if col.lower() in columns_to_keep]]
        if output_crs:
            data = data.to_crs(output_crs)

    return data


def get_transect_points_dict(feature: gpd.GeoDataFrame) -> dict:
    """Returns dict of np.arrays of transect start and end points
    Example
    {
        'usa_CA_0289-0055-NA1': array([[-13820440.53165404,   4995568.65036405],
        [-13820940.93156407,   4995745.1518021 ]]),
        'usa_CA_0289-0056-NA1': array([[-13820394.24579453,   4995700.97802925],
        [-13820900.16320004,   4995862.31860808]])
    }
    Args:
        feature (gpd.GeoDataFrame): clipped transects within roi
    Returns:
        dict: dict of np.arrays of transect start and end points
        of form {
            '<transect_id>': array([[start point],
                        [end point]]),}
    """
    features = []
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode(ignore_index=True)
    # For each linestring portion of feature convert to lat,lon tuples
    lat_lng = feature_exploded.apply(
        lambda row: {str(row.id): np.array(np.array(row.geometry.coords).tolist())},
        axis=1,
    )
    features = list(lat_lng)
    new_dict = {}
    for item in list(features):
        new_dict = {**new_dict, **item}
    return new_dict


def get_cross_distance_df(
    extracted_shorelines: dict, cross_distance_transects: dict
) -> pd.DataFrame:
    transects_csv = {}
    # copy dates from extracted shoreline
    transects_csv["dates"] = extracted_shorelines["dates"]
    # add cross distances for each transect within the ROI
    transects_csv = {**transects_csv, **cross_distance_transects}
    return pd.DataFrame(transects_csv)


def save_transect_intersections(
    save_path: str,
    extracted_shorelines: dict,
    cross_distance_transects: dict,
    filename: str = "transect_time_series.csv",
) -> str:
    cross_distance_df = get_cross_distance_df(
        extracted_shorelines, cross_distance_transects
    )
    filepath = os.path.join(save_path, filename)
    if os.path.exists(filepath):
        print(f"Overwriting:{filepath}")
        os.remove(filepath)
    cross_distance_df.to_csv(filepath, sep=",")
    return filepath


def remove_z_coordinates(geodf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """If the geodataframe has z coordinates in any rows, the z coordinates are dropped.
    Otherwise the original geodataframe is returned.

    Additionally any multi part geometeries will be exploded into single geometeries.
    eg. MutliLineStrings will be converted into LineStrings.
    Args:
        geodf (gpd.GeoDataFrame): geodataframe to check for z-axis

    Returns:
        gpd.GeoDataFrame: original dataframe if there is no z axis. If a z axis is found
        a new geodataframe is returned with z axis dropped.
    """
    if geodf.empty:
        logger.warning(f"Empty geodataframe has no z-axis")
        return geodf

    # if any row has a z coordinate then remove the z_coordinate
    logger.info(f"Has Z axis: {geodf['geometry'].has_z.any()}")
    if geodf["geometry"].has_z.any():

        def remove_z_from_row(row):
            if row.geometry.has_z:
                row.geometry = shapely.ops.transform(
                    lambda x, y, z=None: (x, y), row.geometry
                )
                return row
            else:
                return row

        # Use explode to break multilinestrings in linestrings
        feature_exploded = geodf.explode(ignore_index=True)
        # For each linestring portion of feature convert to lat,lon tuples
        no_z_gdf = feature_exploded.apply(remove_z_from_row, axis=1)
        return no_z_gdf
    else:
        # @debug not sure if this will break everything
        # Use explode to break multilinestrings in linestrings
        return geodf.explode(ignore_index=True)


def create_csv_per_transect(
    roi_id: str,
    save_path: str,
    cross_distance_transects: dict,
    extracted_shorelines_dict: dict,
    filename: str = "_timeseries_raw.csv",
):
    for key in cross_distance_transects.keys():
        df = pd.DataFrame()
        out_dict = dict([])
        # copy shoreline intersects for each transect
        out_dict[key] = cross_distance_transects[key]
        logger.info(
            f"out dict roi_ids columns : {[roi_id for _ in range(len(extracted_shorelines_dict['dates']))]}"
        )
        out_dict["roi_id"] = [
            roi_id for _ in range(len(extracted_shorelines_dict["dates"]))
        ]
        out_dict["dates"] = extracted_shorelines_dict["dates"]
        out_dict["satname"] = extracted_shorelines_dict["satname"]
        logger.info(f"out_dict : {out_dict}")
        df = pd.DataFrame(out_dict)
        df.index = df["dates"]
        df.pop("dates")
        # save to csv file session path
        csv_filename = f"{key}{filename}"
        fn = os.path.join(save_path, csv_filename)
        logger.info(f"Save time series to {fn}")
        if os.path.exists(fn):
            logger.info(f"Overwriting:{fn}")
            os.remove(fn)
        df.to_csv(fn, sep=",")
        logger.info(
            f"ROI: {roi_id}Time-series of the shoreline change along the transects saved as:{fn}"
        )


def save_extracted_shoreline_figures(
    extracted_shorelines: "Extracted_Shoreline", save_path: str
):
    """
    Save extracted shoreline figures to a specified save path.

    The function first constructs the path to the extracted shoreline figures
    and checks if the path exists. If the path exists, it moves the files to a
    new directory specified by save_path.

    :param extracted_shorelines:An Extracted_Shoreline object containing the extracted shorelines and shoreline settings.
    :param save_path: The path where the output figures will be saved.
    """
    data_path = extracted_shorelines.shoreline_settings["inputs"]["filepath"]
    sitename = extracted_shorelines.shoreline_settings["inputs"]["sitename"]
    extracted_shoreline_figure_path = os.path.join(
        data_path, sitename, "jpg_files", "detection"
    )
    logger.info(f"extracted_shoreline_figure_path: {extracted_shoreline_figure_path}")

    if os.path.exists(extracted_shoreline_figure_path):
        dst_path = os.path.join(save_path, "jpg_files", "detection")
        logger.info(f"dst_path : {dst_path }")
        file_utilities.move_files(
            extracted_shoreline_figure_path, dst_path, delete_src=True
        )


def save_extracted_shorelines(
    extracted_shorelines: "Extracted_Shoreline", save_path: str
):
    """
    Save extracted shorelines, settings, and dictionary to their respective files.

    The function saves the following files in the specified save_path:
    - extracted_shorelines.geojson: contains the extracted shorelines as a GeoJSON object.
    - shoreline_settings.json: contains the shoreline settings as JSON data.
    - extracted_shorelines_dict.json: contains the extracted shorelines dictionary as JSON data.

    :param extracted_shorelines: An Extracted_Shoreline object containing the extracted shorelines, shoreline settings, and dictionary.
    :param save_path: The path where the output files will be saved.
    """
    # create a geodataframe of the extracted_shorelines as linestrings
    extracted_shorelines_gdf_lines = extracted_shorelines.create_geodataframe(
        extracted_shorelines.shoreline_settings["output_epsg"],
        output_crs="EPSG:4326",
        geomtype="lines",
    )

    # Save extracted shorelines as a GeoJSON file
    extracted_shorelines.to_file(
        save_path, "extracted_shorelines_lines.geojson", extracted_shorelines_gdf_lines
    )

    # Save extracted shorelines as a GeoJSON file
    extracted_shorelines.to_file(
        save_path, "extracted_shorelines_points.geojson", extracted_shorelines.gdf
    )

    # Save shoreline settings as a JSON file
    extracted_shorelines.to_file(
        save_path,
        "shoreline_settings.json",
        extracted_shorelines.shoreline_settings,
    )

    # Save extracted shorelines dictionary as a JSON file
    extracted_shorelines.to_file(
        save_path,
        "extracted_shorelines_dict.json",
        extracted_shorelines.dictionary,
    )


def stringify_datetime_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Check if any of the columns in a GeoDataFrame have the type pandas timestamp and convert them to string.

    Args:
        gdf: A GeoDataFrame.

    Returns:
        A new GeoDataFrame with the same data as the original, but with any timestamp columns converted to string.
    """
    timestamp_cols = [
        col for col in gdf.columns if pd.api.types.is_datetime64_any_dtype(gdf[col])
    ]

    if not timestamp_cols:
        return gdf

    gdf = gdf.copy()

    for col in timestamp_cols:
        gdf[col] = gdf[col].astype(str)

    return gdf


def create_json_config(inputs: dict, settings: dict, roi_ids: list[str] = []) -> dict:
    """returns config dictionary with the settings, currently selected_roi ids, and
    each of the inputs specified by roi id.
    sample config:
    {
        'roi_ids': ['17','20']
        'settings':{ 'dates': ['2018-12-01', '2019-03-01'],
                    'cloud_thresh': 0.5,
                    'dist_clouds': 300,
                    'output_epsg': 3857,}
        '17':{
            'sat_list': ['L8'],
            'landsat_collection': 'C01',
            'dates': ['2018-12-01', '2019-03-01'],
            'sitename':'roi_17',
            'filepath':'C:\\Home'
        }
        '20':{
            'sat_list': ['L8'],
            'landsat_collection': 'C01',
            'dates': ['2018-12-01', '2019-03-01'],
            'sitename':'roi_20',
            'filepath':'C:\\Home'
        }
    }

    Args:
        inputs (dict): json style dictionary with roi ids at the keys with inputs as values
        settings (dict):  json style dictionary containing map settings
    Returns:
        dict: json style dictionary, config
    """
    if not roi_ids:
        roi_ids = list(inputs.keys())
    config = {**inputs}
    config["roi_ids"] = roi_ids
    config["settings"] = settings
    logger.info(f"config_json: {config}")
    return config


def set_crs_or_initialize_empty(gdf: gpd.GeoDataFrame, epsg_code: str):
    """Set the CRS for the given GeoDataFrame or initialize an empty one."""
    if gdf is not None and not gdf.empty:
        return gdf.to_crs(epsg_code)
    return gpd.GeoDataFrame(geometry=[], crs=epsg_code)


def create_config_gdf(
    rois_gdf: gpd.GeoDataFrame,
    shorelines_gdf: gpd.GeoDataFrame = None,
    transects_gdf: gpd.GeoDataFrame = None,
    bbox_gdf: gpd.GeoDataFrame = None,
    epsg_code: int = None,
) -> gpd.GeoDataFrame:
    """
    Create a concatenated GeoDataFrame from provided GeoDataFrames with a consistent CRS.

    Parameters:
    - rois_gdf (gpd.GeoDataFrame): The GeoDataFrame containing Regions of Interest (ROIs).
    - shorelines_gdf (gpd.GeoDataFrame, optional): The GeoDataFrame containing shorelines. Defaults to None.
    - transects_gdf (gpd.GeoDataFrame, optional): The GeoDataFrame containing transects. Defaults to None.
    - bbox_gdf (gpd.GeoDataFrame, optional): The GeoDataFrame containing bounding boxes. Defaults to None.
    - epsg_code (int, optional): The EPSG code for the desired CRS. If not provided and rois_gdf is non-empty,
      the CRS of rois_gdf will be used. If not provided and rois_gdf is empty, an error will be raised.

    Returns:
    - gpd.GeoDataFrame: A concatenated GeoDataFrame with a consistent CRS, and a "type" column
      indicating the type of each geometry (either "roi", "shoreline", "transect", or "bbox").

    Raises:
    - ValueError: If both epsg_code is None and rois_gdf is None or empty.

    Notes:
    - The function will convert each provided GeoDataFrame to the specified CRS.
    - If any of the input GeoDataFrames is None or empty, it will be initialized as an empty GeoDataFrame
      with the specified CRS.
    """
    # Determine CRS
    if not epsg_code and (rois_gdf is None or rois_gdf.empty):
        raise ValueError(
            "Either provide a valid epsg code or a non-empty rois_gdf to determine the CRS."
        )
    if not epsg_code:
        epsg_code = rois_gdf.crs

    # Dictionary to map gdf variables to their types
    gdfs = {
        "roi": rois_gdf,
        "shoreline": shorelines_gdf,
        "transect": transects_gdf,
        "bbox": bbox_gdf,
    }

    # initialize each gdf
    for gdf_type, gdf in gdfs.items():
        gdfs[gdf_type] = set_crs_or_initialize_empty(gdf, epsg_code)
        gdfs[gdf_type]["type"] = gdf_type

    # Concatenate GeoDataFrames into a single config gdf
    config_gdf = pd.concat(gdfs.values(), ignore_index=True)

    return config_gdf


def get_jpgs_from_data() -> str:
    """Returns the folder where all jpgs were copied from the data folder in coastseg.
    This is where the model will save the computed segmentations."""
    # Data folder location
    src_path = os.path.abspath(os.getcwd() + os.sep + "data")
    if os.path.exists(src_path):
        rename_jpgs(src_path)
        # Create a new folder to hold all the data
        location = os.getcwd()
        name = "segmentation_data"
        # new folder "segmentation_data_datetime"
        new_folder = file_utilities.mk_new_dir(name, location)
        # create subdirectories for each image type
        file_types = ["RGB", "SWIR", "NIR"]
        for file_type in file_types:
            new_path = os.path.join(new_folder, file_type)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            glob_str = (
                src_path
                + str(os.sep + "**" + os.sep) * 2
                + "preprocessed"
                + os.sep
                + file_type
                + os.sep
                + "*.jpg"
            )
            file_utilities.copy_files_to_dst(src_path, new_path, glob_str)
            RGB_path = os.path.join(new_folder, "RGB")
        return RGB_path
    else:
        print("ERROR: Cannot find the data directory in coastseg")
        raise Exception("ERROR: Cannot find the data directory in coastseg")


def save_config_files(
    save_location: str = "",
    roi_ids: list[str] = [],
    roi_settings: dict = {},
    shoreline_settings: dict = {},
    transects_gdf=None,
    shorelines_gdf=None,
    roi_gdf=None,
    epsg_code="epsg:4326",
):
    # save config files
    config_json = create_json_config(roi_settings, shoreline_settings, roi_ids=roi_ids)
    file_utilities.config_to_file(config_json, save_location)
    # save a config geodataframe with the rois, reference shoreline and transects
    if roi_gdf is not None:
        if not roi_gdf.empty:
            epsg_code = roi_gdf.crs
    config_gdf = create_config_gdf(
        rois_gdf=roi_gdf,
        shorelines_gdf=shorelines_gdf,
        transects_gdf=transects_gdf,
        epsg_code=epsg_code,
    )
    file_utilities.config_to_file(config_gdf, save_location)


def rename_jpgs(src_path: str) -> None:
    """Renames all the jpgs in the data directory in coastseg
    Args:
        src_path (str): full path to the data directory in coastseg
    """
    files_renamed = False
    for folder in os.listdir(src_path):
        folder_path = src_path + os.sep + folder
        # Split the folder name at the first _
        folder_id = folder.split("_")[0]
        folder_path = folder_path + os.sep + "jpg_files" + os.sep + "preprocessed"
        jpgs = glob.glob1(folder_path + os.sep, "*jpg")
        # Append folder id to basename of jpg if not already there
        for jpg in jpgs:
            if folder_id not in jpg:
                files_renamed = True
                base, ext = os.path.splitext(jpg)
                new_name = folder_path + os.sep + base + "_" + folder_id + ext
                old_name = folder_path + os.sep + jpg
                os.rename(old_name, new_name)
        if files_renamed:
            print(f"Renamed files in {src_path} ")


def do_rois_filepaths_exist(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if all rois have filepaths that exist
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map
    Returns:
        bool: True if all rois have filepaths that exist
    """
    # by default assume all filepaths exist
    does_filepath_exist = True
    for roi_id in roi_ids:
        filepath = str(roi_settings[roi_id]["filepath"])
        if not os.path.exists(filepath):
            # if filepath does not exist stop checking
            does_filepath_exist = False
            logger.info(f"filepath did not exist{filepath}")
            print("Some ROIs contained filepaths that did not exist")
            break
    logger.info(f"{does_filepath_exist} All rois filepaths exist")
    return does_filepath_exist


def do_rois_have_sitenames(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if all rois have "sitename" with non-empty string
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map

    Returns:
        bool: True if all rois have "sitename" with non-empty string
    """
    # by default assume all sitenames are not empty
    is_sitename_not_empty = True
    for roi_id in roi_ids:
        if roi_settings[roi_id]["sitename"] == "":
            # if sitename is empty means user has not downloaded ROI data
            is_sitename_not_empty = False
            break
    logger.info(f"{is_sitename_not_empty} All rois have non-empty sitenames")
    return is_sitename_not_empty


def were_rois_downloaded(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if rois were downloaded before. False if they have not
    Uses 'sitename' key for each roi to determine if roi was downloaded.
    And checks if filepath were roi is saved is valid
    If each roi's 'sitename' is not empty string returns true
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map

    Returns:
        bool: True means rois were downloaded before
    """
    # by default assume rois were downloaded
    is_downloaded = True
    if roi_settings is None:
        # if rois do not have roi_settings this means they were never downloaded
        is_downloaded = False
    elif roi_settings == {}:
        # if rois do not have roi_settings this means they were never downloaded
        is_downloaded = False
    elif roi_settings != {}:
        all_sitenames_exist = do_rois_have_sitenames(roi_settings, roi_ids)
        all_filepaths_exist = do_rois_filepaths_exist(roi_settings, roi_ids)
        is_downloaded = all_sitenames_exist and all_filepaths_exist
    # print correct message depending on whether ROIs were downloaded
    if is_downloaded:
        logger.info(f"Located previously downloaded ROI data.")
    elif is_downloaded == False:
        print(
            "Did not locate previously downloaded ROI data. To download the imagery for your ROIs click Download Imagery"
        )
        logger.info(
            f"Did not locate previously downloaded ROI data. To download the imagery for your ROIs click Download Imagery"
        )
    return is_downloaded


def create_roi_settings(
    settings: dict,
    selected_rois: dict,
    filepath: str,
    date_str: str = "",
) -> dict:
    """returns a dict of settings for each roi with roi id as the key.
    Example:
    "2": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_2_datetime10-19-22__04_00_34",
            "filepath": "C:\\CoastSeg\\data",
            "roi_id": "2",
            "polygon": [
                [
                    [-124.16930255115336, 40.8665390046026],
                    [-124.16950858759564, 40.878247531017706],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.1538792781699, 40.8666943403763],
                    [-124.16930255115336, 40.8665390046026],
                ]
            ],
            "landsat_collection": "C01",
        },
        "3": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sat_list": ["L8"],
            "sitename": "ID_3_datetime10-19-22__04_00_34",
            "filepath": "C:\\CoastSeg\\data",
            "roi_id": "3",
            "polygon": [
                [
                    [-124.16950858759564, 40.878247531017706],
                    [-124.16971474532464, 40.88995603272874],
                    [-124.15428603840094, 40.890111496009816],
                    [-124.15408259844114, 40.878402930533994],
                    [-124.16950858759564, 40.878247531017706],
                ]
            ],
            "landsat_collection": "C01",
        },

    Args:
        settings (dict): settings from coastseg_map.
        Must have keys ["sat_list","landsat_collection","dates"]
        selected_rois (dict): geojson dict of rois selected
        filepath (str): file path to directory to hold roi data
        date_str (str, optional): datetime formatted string. Defaults to "".

    Returns:
        dict: settings for each roi with roi id as the key
    """

    roi_settings = {}
    sat_list = settings["sat_list"]
    landsat_collection = settings["landsat_collection"]
    dates = settings["dates"]
    for roi in selected_rois["features"]:
        roi_id = str(roi["properties"]["id"])
        sitename = (
            "" if date_str == "" else "ID_" + str(roi_id) + "_datetime" + date_str
        )
        polygon = roi["geometry"]["coordinates"]
        inputs_dict = {
            "dates": dates,
            "sat_list": sat_list,
            "roi_id": roi_id,
            "polygon": polygon,
            "landsat_collection": landsat_collection,
            "sitename": sitename,
            "filepath": filepath,
        }
        roi_settings[roi_id] = inputs_dict
    return roi_settings


def scale(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """returns resized matrix with shape(rows,cols)
        for 2d discrete labels
        for resizing 2d integer arrays
    Args:
        im (np.ndarray): 2d matrix to resize
        nR (int): number of rows to resize 2d matrix to
        nC (int): number of columns to resize 2d matrix to

    Returns:
        np.ndarray: resized matrix with shape(rows,cols)
    """
    src_rows = len(matrix)  # source number of rows
    src_cols = len(matrix[0])  # source number of columns
    tmp = [
        [
            matrix[int(src_rows * r / rows)][int(src_cols * c / cols)]
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    return np.array(tmp).reshape((rows, cols))


def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    Code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn
