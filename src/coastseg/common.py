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
from typing import List
from datetime import datetime, timezone
from typing import Dict, List

# Third-party imports
import ee
from google.auth import exceptions as google_auth_exceptions
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
from shapely.geometry import MultiPoint, LineString

# Specific classes/functions from modules
from typing import Callable, List, Optional, Union, Dict, Set, Any, Tuple

# Internal dependencies imports
from coastseg import exceptions
from coastseg.validation import find_satellite_in_filename
from coastseg import file_utilities
from coastseg.exceptions import InvalidGeometryType
# widget icons from https://fontawesome.com/icons/angle-down?s=solid&f=classic

# Logger setup
logger = logging.getLogger(__name__)

def update_config(config_json: dict, roi_settings: dict) -> dict:
    """
    Update the configuration JSON with the provided ROI settings.

    Args:
        config_json (dict): The original configuration JSON.
        roi_settings (dict): The ROI settings to be updated.

    Returns:
        dict: The updated configuration JSON.
    """
    for roi_id, settings in roi_settings.items():
        if roi_id in config_json:
            config_json[roi_id].update(settings)
    return config_json


def update_downloaded_configs(roi_settings: dict, roi_ids: list =None):
    """
    Update the downloaded configuration files for the specified ROI(s).
    Args:
        roi_settings (dict, optional): Dictionary containing the ROI settings. Defaults to None.
            ROI settings should contain the ROI IDs as the keys and a dictionary of settings as the values.
            Each ROI ID should have the following keys: "dates", "sitename", "polygon", "roi_id", "sat_list", "landsat_collection", "filepath"
        roi_ids (list, optional): List of ROI IDs to update. Defaults to None.
    """
    if not isinstance(roi_settings, dict):
        raise ValueError("Invalid roi_settings provided.")
    
    if not roi_ids:
        roi_ids = list(roi_settings.keys())
    if isinstance(roi_ids, str):
        roi_ids = [roi_ids]
    
    for roi_id in roi_ids:
        try:
            # read the settings for the current ROI
            settings = roi_settings.get(roi_id,{})
            if not settings:
                logging.warning(f"No settings found for ROI {roi_id}. Skipping.")
                continue

            config_path = os.path.join(settings["filepath"], settings["sitename"], "config.json")
            
            if not os.path.exists(config_path):
                logging.warning(f"Config file not found for ROI {roi_id}. Skipping.")
                continue

            # load the current contents of the config.json file
            config_json = file_utilities.read_json_file(config_path)
            # Update the ROI data for each ROI in config.json
            updated_config = update_config(config_json, roi_settings)
            file_utilities.config_to_file(updated_config, config_path)
            logging.info(f"Successfully updated config for ROI {roi_id} at {config_path}")
        except IOError as e:
            logging.error(f"Failed to update config for ROI {roi_id}: {e}")

def extract_roi_settings(json_data: dict,fields_of_interest: set = set(),roi_ids: list = None) -> dict:
    """
    Extracts the settings for regions of interest (ROI) from the given JSON data.
    Overwrites the filepath attribute for each ROI with the data_path provided.
    Args:
        json_data (dict): The JSON data containing ROI information.
        data_path (str): The path to the data directory.
        fields_of_interest (set, optional): A set of fields to include in the ROI settings.
            Defaults to an empty set.
    Returns:
        dict: A dictionary containing the ROI settings, where the keys are ROI IDs and
            the values are dictionaries containing the fields of interest for each ROI.
    """
    if not fields_of_interest:
        fields_of_interest = {
                "dates",
                "sitename",
                "polygon",
                "roi_id",
                "sat_list",
                "landsat_collection",
                "filepath",
            }
    if not roi_ids:
        roi_ids = json_data.get("roi_ids", [])
    roi_settings = {}
    for roi_id in roi_ids:
        # create a dictionary containing the fields of interest for the ROI with the roi_id
        roi_data = extract_roi_data(json_data, roi_id, fields_of_interest)
        roi_settings[str(roi_id)] = roi_data
    return roi_settings

def update_roi_settings(roi_settings, key, value):
    """
    Updates the settings for a region of interest (ROI) in the given ROI settings dictionary.

    Args:
        roi_settings (dict): A dictionary containing the ROI settings.
        key (str): The key of the ROI settings to update.
        value (Any): The new value for the specified key.

    Returns:
        dict: The updated ROI settings dictionary.

    """
    for roi_id, settings in roi_settings.items():
        if key in settings:
            settings[key] = value
    return roi_settings

def process_roi_settings(json_data, data_path)->dict:
    """
    Process the ROI settings from the given JSON data and update the filepath to be the data_path.

    Args:
        json_data (dict): The JSON data containing ROI settings.
        data_path (str): The path to the data directory.

    Returns:
        dict: A dictionary mapping ROI IDs to their extracted settings with updated filepath.
    """
    roi_ids = json_data.get("roi_ids", [])
    roi_settings = extract_roi_settings(json_data, roi_ids=roi_ids)
    roi_settings = update_roi_settings(roi_settings, 'filepath', data_path)
    return roi_settings

def get_missing_roi_dirs(roi_settings: dict, roi_ids: list = None) -> dict:
    """
    Get the missing ROI directories based on the provided ROI settings and data path.

    Args:
        roi_settings (dict): A dictionary containing ROI settings.
        roi_ids (list, optional): A list of ROI IDs to check. If not provided, all ROIs in roi_settings are checked. Defaults to None.

    Returns:
        dict: A dictionary containing the missing ROI directories, where the key is the ROI ID and the value is the sitename.
    """
    missing_directories = {}
    if roi_settings == {}:
        return missing_directories

    # If roi_ids is not provided, check all ROIs in roi_settings
    if roi_ids is None:
        roi_ids = roi_settings.keys()

    for roi_id in roi_ids:
        item = roi_settings.get(roi_id, {})
        sitename = item.get("sitename", "")
        filepath = item.get("filepath", "")
        roi_path = os.path.join(filepath, sitename)

        if not os.path.exists(roi_path):
            missing_directories[roi_id] = sitename

    return missing_directories

def initialize_gee(
    auth_mode: str = "localhost",
    print_mode: bool = True,
    auth_args: dict = {},
    **kwargs,
):
    """
    Initialize Google Earth Engine (GEE). If initialization fails due to authentication issues,prompt the user to authenticate and try again.

    Arguments:
    -----------
        auth_mode (str, optional): The authentication mode, can be one of localhost.
                Note: gcloud method of authentication is not supported.
                Note: colab method of authentication is not supported.
                See https://developers.google.com/earth-engine/guides/auth for more details. Defaults to localhost.
        print_mode (bool, optional): Whether to print messages to the console. Defaults to True.
        auth_args (dict, optional): Additional authentication parameters for aa.Authenticate(). Defaults to {}.
        kwargs (dict, optional): Additional parameters for ee.Initialize().

    """
    auth_args = {"auth_mode": auth_mode}
    # update auth_args
    auth_args.update(kwargs)

    if auth_mode == "colab":
        raise ValueError("Colab authentication is not supported.")
    elif auth_mode == "gcloud":
        raise ValueError("GCloud authentication is not supported.")
    elif auth_mode == "notebook":
        raise ValueError("Notebook authentication is not supported.")

    try:
        if print_mode:
            print("Initializing Google Earth Engine...\n")
        ee.Initialize(**kwargs)
        if print_mode:
            print("Google Earth Engine initialized successfully.\n")
        return
    except google_auth_exceptions.RefreshError:
        print("Please refresh your Google authentication token.\n")
        ee.Authenticate(**auth_args)
    except ee.EEException:
        print("Please authenticate with Google Earth Engine:\n")
        ee.Authenticate(**auth_args)
    except FileNotFoundError:
        print(
            "Credentials file not found. Please authenticate with Google Earth Engine:\n"
        )
        ee.Authenticate(**auth_args)

    # Try to initialize again after authentication
    try:
        if print_mode:
            print("Attempt 2: Initializing Google Earth Engine...\n")
        ee.Initialize(**kwargs)
        if print_mode:
            print("Attempt 2: Google Earth Engine initialized successfully.\n")
    except Exception as e:
        raise Exception(f"Failed to initialize Google Earth Engine:\n {e}")


def create_new_config(roi_ids: list, settings: dict, roi_settings: dict) -> dict:
    """
    Creates a new configuration dictionary by combining the given settings and ROI settings.

    Arguments:
    -----------
    roi_ids: list
        A list of ROI IDs to include in the new configuration.
    settings: dict
        A dictionary containing general settings for the configuration.
    roi_settings: dict
        A dictionary containing ROI-specific settings for the configuration.
        example:
        {'example_roi_id': {'dates':[]}

    Returns:
    -----------
    new_config: dict
        A dictionary containing the combined settings and ROI settings, as well as the ROI IDs.
    """
    new_config = {
        "settings": {},
        "roi_ids": [],
    }
    if isinstance(roi_ids, str):
        roi_ids = [roi_ids]
    if not all(roi_id in roi_settings.keys() for roi_id in roi_ids):
        raise ValueError(f"roi_ids {roi_ids} not in roi_settings {roi_settings.keys()}")
    new_config = {**new_config, **roi_settings}
    new_config["roi_ids"].extend(roi_ids)
    new_config["settings"] = settings
    return new_config


def update_transect_time_series(
    filepaths: List[str], dates_list: List[datetime]
) -> None:
    """
    Updates a series of CSV files by removing rows based on certain dates.

    :param filepaths: A list of file paths to the CSV files.
    :param dates_list: A list of datetime objects representing the dates to be filtered out.
    :return: None
    """
    for filepath in filepaths:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)

        # Format the dates to match the format in the CSV file
        formatted_dates = [
            date.strftime("%Y-%m-%d %H:%M:%S+00:00") for date in dates_list
        ]
        # Keep only the rows where the 'dates' column isn't in the list of formatted dates
        df = df[~df["dates"].isin(formatted_dates)]
        # Write the updated DataFrame to the same CSV file
        df.to_csv(filepath, index=False)


def extract_dates_and_sats(
    selected_items: List[str],
) -> Tuple[List[datetime], List[str]]:
    """
    Extract the dates and satellite names from a list of selected items.

    Args:
        selected_items: A list of strings, where each string is in the format "satname_dates".

    Returns:
        A tuple of two lists: the first list contains datetime objects corresponding to the dates in the selected items,
        and the second list contains the satellite names in the selected items.
    """
    dates_list = []
    sat_list = []
    for criteria in selected_items:
        satname, dates = criteria.split("_")
        sat_list.append(satname)
        dates_list.append(
            datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        )
    return dates_list, sat_list


def transform_data_to_nested_arrays(
    data_dict: Dict[str, Union[List[Union[int, float, np.ndarray]], np.ndarray]]
) -> Dict[str, np.ndarray]:
    """
    Convert a dictionary of data to a new dictionary with nested NumPy arrays.

    Args:
        data_dict: A dictionary of data, where each value is either a list of integers, floats, or NumPy arrays, or a NumPy array.

    Returns:
        A new dictionary with the same keys as `data_dict`, where each value is a NumPy array or a nested NumPy array.

    Raises:
        TypeError: If `data_dict` is not a dictionary, or if any value in `data_dict` is not a list or NumPy array.
    """
    transformed_dict = {}
    for key, items in data_dict.items():
        if any(isinstance(element, np.ndarray) for element in items):
            nested_array = np.empty(len(items), dtype=object)
            for index, array_element in enumerate(items):
                nested_array[index] = array_element
            transformed_dict[key] = nested_array
        else:
            transformed_dict[key] = np.array(items)
    return transformed_dict


def process_data_input(data):
    """
    Process the data input and transform it to nested arrays.

    Parameters:
    data (dict or str): The data input to process. If data is a string, it is assumed to be the full path to the JSON file.

    Returns:
    dict: The processed data as nested arrays.
    """
    # Determine if data is a dictionary or a file path
    if isinstance(data, dict):
        data_dict = data
    elif isinstance(data, str):
        # Load data from the JSON file
        if os.path.exists(data):
            data_dict = file_utilities.load_data_from_json(data)
        else:
            return None
    else:
        raise TypeError("data must be either a dictionary or a string file path.")

    # Transform data to nested arrays
    new_dict = transform_data_to_nested_arrays(data_dict)
    return new_dict


def update_extracted_shorelines_dict_transects_dict(
    session_path, filename, dates_list, sat_list
):
    json_file = os.path.join(session_path, filename)
    if os.path.exists(json_file) and os.path.isfile(json_file):
        # read the data from the json file
        data = file_utilities.load_data_from_json(json_file)
        # processes the data into nested arrays
        extracted_shorelines_dict = process_data_input(data)
        if extracted_shorelines_dict is not None:
            # Get the indexes of the selected items in the extracted_shorelines_dict
            selected_indexes = get_selected_indexes(
                extracted_shorelines_dict, dates_list, sat_list
            )
            # attempt to delete the selected indexes from the "transect_cross_distances.json"
            transect_cross_distances_path = os.path.join(
                session_path, "transects_cross_distances.json"
            )
            # if the transect_cross_distances.json exists then delete the selected indexes from it
            if os.path.exists(transect_cross_distances_path) and os.path.isfile(
                transect_cross_distances_path
            ):
                transects_dict = process_data_input(transect_cross_distances_path)
                if transects_dict is not None:
                    # Delete the selected indexes from the transects_dict
                    transects_dict = delete_selected_indexes(
                        transects_dict, selected_indexes
                    )
                    file_utilities.to_file(
                        transects_dict, transect_cross_distances_path
                    )

            # Delete the selected indexes from the extracted_shorelines_dict
            extracted_shorelines_dict = delete_selected_indexes(
                extracted_shorelines_dict, selected_indexes
            )
            file_utilities.to_file(extracted_shorelines_dict, json_file)


def delete_selected_indexes(input_dict, selected_indexes):
    """
    Delete the selected indexes from the transects_dict.

    Parameters:
    input_dict (dict): The transects dictionary to modify.
    selected_indexes (list): The indexes to delete.

    Returns:
    dict: The modified transects dictionary.
    """
    if not selected_indexes:
        return input_dict
    for key in input_dict.keys():
        was_list = False
        if isinstance(input_dict[key], list):
            was_list = True
        if any(isinstance(element, np.ndarray) for element in input_dict[key]):
            nested_array = np.empty(len(input_dict[key]), dtype=object)
            for index, array_element in enumerate(input_dict[key]):
                nested_array[index] = array_element
            input_dict[key] = nested_array
            # now delete the selected indexes
            input_dict[key] = np.delete(input_dict[key], selected_indexes)
            # then transform back to into a list
            if was_list == True:
                input_dict[key] = input_dict[key].tolist()
        else:
            input_dict[key] = np.delete(input_dict[key], selected_indexes)
    return input_dict


def load_settings(
    filepath: str = "",
    keys: set = (
        "model_session_path",
        "apply_cloud_mask",
        "image_size_filter",
        "pan_off",
        "save_figure",
        "adjust_detection",
        "check_detection",
        "landsat_collection",
        "sat_list",
        "dates",
        "sand_color",
        "cloud_thresh",
        "cloud_mask_issue",
        "min_beach_area",
        "min_length_sl",
        "output_epsg",
        "sand_color",
        "pan_off",
        "max_dist_ref",
        "dist_clouds",
        "percent_no_data",
        "max_std",
        "min_points",
        "along_dist",
        "max_range",
        "min_chainage",
        "multiple_inter",
        "prc_multiple",
    ),
    new_settings: dict={},
):
    """
    Loads settings from a JSON file and applies them to the object.
    Args:
        filepath (str, optional): The filepath to the JSON file containing the settings. Defaults to an empty string.
        keys (list or set, optional): A list of keys specifying which settings to load from the JSON file. If empty, no settings are loaded. Defaults to a set with the following
                                                    "sat_list",
                                                    "dates",
                                                    "cloud_thresh",
                                                    "cloud_mask_issue",
                                                    "min_beach_area",
                                                    "min_length_sl",
                                                    "output_epsg",
                                                    "sand_color",
                                                    "pan_off",
                                                    "max_dist_ref",
                                                    "dist_clouds",
                                                    "percent_no_data",
                                                    "max_std",
                                                    "min_points",
                                                    "along_dist",
                                                    "max_range",
                                                    "min_chainage",
                                                    "multiple_inter",
                                                    "prc_multiple".
        new_settings(dict, optional): A dictionary containing new settings to apply to the object. Defaults to an empty dictionary.
    Returns:
        None
    """
    # Convert keys to a list if a set is passed
    if isinstance(keys, set):
        keys = list(keys)
    if filepath:
        new_settings = file_utilities.read_json_file(filepath, raise_error=False)
        logger.info(
            f"all of new settings read from file : {filepath} \n {new_settings.keys()}"
        )
    elif new_settings:
        logger.info(f"all of new settings read from dict : {new_settings.keys()}") 
    # if no keys are passed then use all of the keys in the settings file
    if not keys:
        keys = new_settings.keys()
    # filter the settings to keep only the keys passed
    filtered_settings = {k: new_settings[k] for k in keys if k in new_settings}
    # read the nested settings located in the sub dictionary "settings" and keep only the keys passed
    nested_settings = new_settings.get("settings", {})
    nested_settings = {k: nested_settings[k] for k in keys if k in nested_settings}
    logger.info(
        f"all of new nested settings read from file : {filepath} \n {new_settings.keys()}"
    )
    # combine the settings into one dictionary WARNING this could overwrite items in both settings
    filtered_settings.update(**nested_settings)
    return filtered_settings


def remove_matching_rows(gdf: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
    """
    Remove rows from a GeoDataFrame that match ALL the columns and items specified in the keyword arguments.

    Each of the keyword argument should be a column name in the GeoDataFrame with the values
    to filter in the given column.

    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame.
    **kwargs: Keyword arguments representing column names and items to match.

    Returns:
    GeoDataFrame: The modified GeoDataFrame with matching rows removed.
    """

    # Initialize a mask with all True values
    combined_mask = pd.Series([True] * len(gdf))

    for column_name, items_list in kwargs.items():
        # Ensure the column exists in the DataFrame
        if column_name not in gdf.columns:
            raise ValueError(f"Column '{column_name}' not found in GeoDataFrame")

        # Create a mask for each condition and combine them using logical AND
        condition_mask = pd.Series([False] * len(gdf))
        # Iterate over the items in the list
        for item in items_list:
            # creates a mask where the column value is equal to the item and the condition mask is True, meaning previous conditions were met
            condition_mask = condition_mask | (gdf[column_name] == item)
        # Combine the condition mask with the combined mask with a logical AND
        combined_mask = combined_mask & condition_mask

        # Convert datetime columns to strings
        if pd.api.types.is_datetime64_any_dtype(gdf[column_name]) or isinstance(
            gdf[column_name].dtype, pd.DatetimeTZDtype
        ):
            gdf[column_name] = gdf[column_name].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Drop the rows that match the combined criteria
    gdf = gdf.drop(gdf[combined_mask].index)

    return gdf


def get_selected_indexes(
    data_dict: Dict[str, Union[List[Any], pd.Series]],
    dates_list: List[Union[str, pd.Timestamp]],
    sat_list: List[str],
) -> List[int]:
    """
    Retrieve indexes of rows in a dictionary that match specified dates and satellite names.

    This function accepts a dictionary containing at least two keys: 'dates' and 'satname'.
    It then returns a list of indexes where the dates and satellite names match those provided in
    the dates_list and sat_list respectively.

    Parameters:
    - data_dict (Dict[str, Union[List[Any], pd.Series]]): The dictionary containing data arrays.
                                                          Expected keys are 'dates' and 'satname'.
                                                          If the keys are absent, they will be set with empty lists.
    - dates_list (List[Union[str, pd.Timestamp]]): A list containing dates to match against.
    - sat_list (List[str]): A list containing satellite names to match against.

    Returns:
    - List[int]: A list of integer indexes where the 'dates' and 'satname' in the data_dict
                 match the provided lists. Returns an empty list if no matches are found or if the data_dict is empty.

    Examples:
    >>> data = {'dates': ['2021-01-01', '2021-01-02'], 'satname': ['sat1', 'sat2']}
    >>> get_selected_indexes(data, ['2021-01-01'], ['sat1'])
    [0]
    """
    if not data_dict:
        return []
    data_dict.setdefault("dates", [])
    data_dict.setdefault("satname", [])
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data_dict)

    # Initialize an empty list to store selected indexes
    selected_indexes = []

    # Iterate over dates and satellite names, and get the index of the first matching row
    for date, sat in zip(dates_list, sat_list):
        match = df[(df["dates"] == date) & (df["satname"] == sat)]
        if not match.empty:
            selected_indexes.append(match.index[0])

    return selected_indexes


def save_new_config(path: str, roi_ids: list, destination: str) -> dict:
    """Save a new config file to a path.

    Args:
        path (str): the path to read the original config file from
        roi_ids (list): a list of roi_ids to include in the new config file
        destination (str):the path to save the new config file to
    """
    with open(path) as f:
        config = json.load(f)

    if isinstance(roi_ids, str):
        roi_ids = [roi_ids]

    roi_settings = {}
    for roi_id in roi_ids:
        if roi_id in config.keys():
            roi_settings[roi_id] = config[roi_id]

    new_config = create_json_config(roi_settings, config["settings"], roi_ids)
    with open(destination, "w") as f:
        json.dump(new_config, f)


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


def drop_dates_from_csv(
    file_pattern: str,
    session_path: str,
    dates_list: list[datetime],
    column_name: str = "dates",
):
    """
    Drops rows from CSV files that match a given file pattern and have dates in a specified list.

    Args:
        file_pattern (str): The pattern to match for the CSV files.
        session_path (str): The path to the directory containing the CSV files.
        dates_list (list[datetime]): The list of datetime objects representing the dates to drop.
        column_name (str, optional): The name of the column containing the dates. Defaults to "dates".

    Raises:
        AssertionError: If any element in the dates_list is not a datetime object.

    Returns:
        None
    """
    # assert that datetime objects are passed
    assert all(
        isinstance(date, datetime) for date in dates_list
    ), "dates_list must contain datetime objects"
    formatted_dates = [date.strftime("%Y-%m-%d %H:%M:%S+00:00") for date in dates_list]

    # Get a list of files that match the file pattern in the directory
    # Compile the regex pattern
    # Filter files based on the regex pattern
    matched_files = glob.glob(session_path + f"/*{file_pattern}*")
    # Loop through each file in the matched files list
    for file in matched_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Convert the column to datetime objects
        df[column_name] = pd.to_datetime(df[column_name])

        # Drop rows where the column is in the list of formatted dates
        df = df[~df[column_name].isin(formatted_dates)]

        # Write the updated DataFrame to the same CSV file
        df.to_csv(file, index=False)


def delete_jpg_files(
    dates_list: List[datetime], sat_list: List[str], jpg_path: str
) -> None:
    """
    Delete JPEG files based on the given dates and satellite list.

    Args:
        dates_list (List[datetime]): A list of datetime objects representing the dates.
        sat_list (list): A list of satellite names.
        jpg_path (str): The path to the directory containing the JPEG files.

    Returns:
        None
    """
    # assert that datetime objects are passed
    assert all(
        isinstance(date, datetime) for date in dates_list
    ), "dates_list must contain datetime objects"
    # Format the dates in dates_list as strings
    formatted_dates = [date.strftime("%Y-%m-%d-%H-%M-%S") for date in dates_list]

    # Get a list of all JPEG files in jpg_path
    jpg_files = set(os.listdir(jpg_path))

    # Create a list of filenames to delete
    delete_list = [
        date + "_" + sat + ".jpg"
        for date, sat in zip(formatted_dates, sat_list)
        if (date + "_" + sat + ".jpg") in jpg_files
    ]

    # Loop through each filename in the delete list
    for filename in delete_list:
        # Construct the full file path by joining the directory path with the filename
        file_path = os.path.join(jpg_path, filename)
        if os.path.exists(file_path):
            # Use the os.remove function to delete the file
            os.remove(file_path)


def filter_partial_images(
    roi_gdf: gpd.geodataframe,
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


def get_roi_area(gdf: gpd.geodataframe) -> float:
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
        logger.error(f"Unable to extract satellite name from filename: {filename}")
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
        "L5": 15,  # coastsat modifies the per pixel resolution from 30m to 15m for L5
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
        img_area = calculate_image_area(filepath, pixel_size_per_satellite[satname])
        if img_area < min_area or (max_area is not None and img_area > max_area):
            bad_files.append(file)

    bad_files = list(map(lambda s: os.path.join(directory, s), bad_files))
    # move the bad files to the bad folder
    file_utilities.move_files(bad_files, output_directory)
    return bad_files  # Optionally return the list of bad files


def calculate_image_area(filepath: str, pixel_size: int) -> float:
    """
    Calculate the area of an image in square kilometers.

    Args:
        filepath (str): The path to the image file.
        pixel_size (int): The size of a pixel in the image in meters.

    Returns:
        float: The area of the image in square kilometers.
    """
    with Image.open(filepath) as img:
        width, height = img.size
        img_area = width * pixel_size * height * pixel_size
        img_area /= 1e6  # convert to square kilometers
    return img_area


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
        file_extension="_timeseries_raw.csv",
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
        if feature_html.value == "":
            container_content.children = [msg]
        elif feature_html.value != "":
            container_content.children = [feature_html]
        container_header.children = [title, collapse_button]
        container.children = [container_header, container_content]

    def collapse_click(change: dict):
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
    box_width: str = "60%",
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
        f"<div style='max-height: 250px; overflow-x: hidden; overflow-y:  auto; text-align: center;'>"
        f"<span style='color: red'>⚠️</span>{msg}"
        f"</div>"
    )
    instructions_msg = HTML(
        f"<div style='max-height: 210px; overflow-x: hidden; overflow-y:  auto; text-align: center;'>"
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

    # make the height of the vbox
    # create vertical box to hold title and msg
    warning_content = VBox(
        [warning_title, warning_msg, instructions_msg, close_button],
        layout=Layout(width=msg_width, max_width="95%",padding='0px 0px 10px 0px',margin='4px 4px 4px 4px'),
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
        layout=Layout(width=box_width, height='100%', border="4px solid red"),
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
            logger.error(f"Error {r.status_code}. DownloadError: {save_path} {r}")
            raise exceptions.DownloadError(os.path.basename(save_path))
        if r.status_code == 429:
            logger.error(f"Error {r.status_code}.DownloadError: {save_path} {r}")
            raise Exception(
                "Zenodo has denied the request. You may have requested too many files at once."
            )
        if r.status_code != 200:
            logger.error(f"Error {r.status_code}. DownloadError: {save_path} {r}")
            raise exceptions.DownloadError(os.path.basename(save_path))
        # check header to get content length, in bytes
        content_length = r.headers.get("Content-Length")
        if content_length:
            total_length = int(content_length)
        else:
            logger.warning("Content length not found in response headers")
            total_length = None

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


def convert_linestrings_to_multipoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert LineString geometries in a GeoDataFrame to MultiPoint geometries.
    Args:
    - gdf (gpd.GeoDataFrame): The input GeoDataFrame.
    Returns:
    - gpd.GeoDataFrame: A new GeoDataFrame with MultiPoint geometries. If the input GeoDataFrame
                        already contains MultiPoints, the original GeoDataFrame is returned.
    """

    # Check if the gdf already contains MultiPoints
    if any(gdf.geometry.type == "MultiPoint"):
        return gdf

    def linestring_to_multipoint(linestring):
        if isinstance(linestring, LineString):
            return MultiPoint(linestring.coords)
        return linestring

    # Convert each LineString to a MultiPoint
    gdf["geometry"] = gdf["geometry"].apply(linestring_to_multipoint)

    return gdf


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


def extract_roi_by_id(gdf: gpd.geodataframe, roi_id: int) -> gpd.geodataframe:
    """Returns geodataframe with a single ROI whose id matches roi_id.
       If roi_id is None returns gdf

    Args:
        gdf (gpd.geodataframe): ROI geodataframe to extract ROI with roi_id from
        roi_id (int): id of the ROI to extract
    Raises:
        exceptions.Id_Not_Found: if id doesn't exist in ROI's geodataframe or self.rois.gdf is empty
    Returns:
        gpd.geodataframe: ROI with id matching roi_id
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


def extract_roi_data(json_data: dict, roi_id: str, fields_of_interest: list = None):
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


def extract_fields(data: dict, key=None, fields_of_interest:list=None)->dict:
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
    # extract the data from a sub dictionary with a specified key if it exists
    if key and key in data:
        for field in fields_of_interest:
            if field in data[key]:
                extracted_data[field] = data[key][field]
    else:  # extract all the fields of interest from the data
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


def get_transect_points_dict(feature: gpd.geodataframe) -> dict:
    """Returns dict of np.arrays of transect start and end points
    Example
    {
        'usa_CA_0289-0055-NA1': array([[-13820440.53165404,   4995568.65036405],
        [-13820940.93156407,   4995745.1518021 ]]),
        'usa_CA_0289-0056-NA1': array([[-13820394.24579453,   4995700.97802925],
        [-13820900.16320004,   4995862.31860808]])
    }
    Args:
        feature (gpd.geodataframe): clipped transects within roi
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
    """
    Creates a DataFrame from extracted shorelines and cross distance transects by
    getting the dates from extracted shorelines and saving it to the as the intersection time for each extracted shoreline
    for each transect

    Parameters:
    extracted_shorelines : dict
        A dictionary containing the extracted shorelines. It must have a "dates" key with a list of dates.
    cross_distance_transects : dict
        A dictionary containing the transects and the cross distance where the extracted shorelines intersected it. The keys are transect names and the values are lists of cross distances.
        eg.
        {  'tranect 1': [1,2,3],
            'tranect 2': [4,5,6],
        }
    Returns:
    DataFrame
        A DataFrame where each column is a transect from cross_distance_transects and the "dates" column from extracted_shorelines. Each row corresponds to a date and contains the cross distances for each transect on that date.
    """
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
    """
    Saves the saves the dates from the extracted shorelines to the dictionart containing the cross distance transect intersections to a CSV file.

    This function processes intersection data between shorelines and transects, removing columns with all NaN values.
    It then saves the processed data to a CSV file at the specified path.

    Args:
    - save_path (str): The directory path where the CSV file will be saved.
    - extracted_shorelines (dict): A dictionary containing shoreline data.
    - cross_distance_transects (dict): A dictionary containing transect data with cross-distance measurements.
    - filename (str, optional): The name of the CSV file to be saved. Default is "transect_time_series.csv".

    Returns:
    - str: The full file path of the saved CSV file.

    The function first combines the shoreline and transect data into a DataFrame and then removes any columns
    that contain only NaN values before saving to CSV.
    """
    cross_distance_df = get_cross_distance_df(
        extracted_shorelines, cross_distance_transects
    )
    cross_distance_df.dropna(axis="columns", how="all", inplace=True)
    filepath = os.path.join(save_path, filename)
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
    file_extension: str = "_timeseries_raw.csv",
) -> None:
    """
    Generates CSV files from transect and shoreline data.

    For each transect in cross_distance_transects, this function creates a CSV file if the transect contains
    non-NaN values. The CSV includes dates, transect data, region of interest ID, and satellite name.

    Args:
    - roi_id (str): ID for the region of interest.
    - save_path (str): Path to save CSV files.
    - cross_distance_transects (dict): Transect data with cross-distance measurements.
    - extracted_shorelines_dict (dict): Contains 'dates' and 'satname'.
    - file_extension (str, optional): File extension for CSV files. Default is "_timeseries_raw.csv".

    Notes:
    - CSV files are named using transect keys and file_extension.
    - Transects with only NaN values are skipped.
    """
    for key, transect in cross_distance_transects.items():
        if pd.notna(transect).any():  # Check if there's any non-NaN value
            # Create DataFrame directly
            df = pd.DataFrame(
                {
                    "dates": extracted_shorelines_dict["dates"],
                    key: transect,
                    "roi_id": [roi_id] * len(extracted_shorelines_dict["dates"]),
                    "satname": extracted_shorelines_dict["satname"],
                },
                index=extracted_shorelines_dict["dates"],
            )
            # Save to csv file
            fn = f"{key}{file_extension}"
            file_path = os.path.join(save_path, fn)
            df.to_csv(
                file_path, sep=",", index=False
            )  # Set index=False if you don't want 'dates' as index in CSV


def move_report_files(
    settings: dict, dest: str, filename_pattern="extract_shorelines*.txt"
):
    """
    Move report files matching a specific pattern from the source directory to the destination.

    :param settings: Dictionary containing 'filepath' and 'sitename'.
    :param dest: The destination path where the report files will be moved.
    :param filename_pattern: Pattern of the filenames to search for, defaults to 'extract_shorelines*.txt'.
    """
    # Attempt to get the data_path and sitename
    filepath = settings.get("filepath") or settings.get("inputs", {}).get("filepath")
    sitename = settings.get("sitename") or settings.get("inputs", {}).get("sitename")

    # Check if data_path and sitename were successfully retrieved
    if not filepath or not sitename:
        logger.error("Data path or sitename not found in settings.")
        return

    # Construct the pattern to match files
    pattern = os.path.join(filepath, sitename, filename_pattern)
    matching_files = glob.glob(pattern)

    # Check if there are files to move
    if not matching_files:
        logger.warning(f"No files found matching the pattern: {pattern}")
        return

    # Move the files
    try:
        file_utilities.move_files(matching_files, dest, delete_src=True)
        logger.info(f"Files moved successfully to {dest}")
    except Exception as e:
        logger.error(f"Error moving files: {e}")


def save_extracted_shoreline_figures(settings: dict, save_path: str):
    """
    Save extracted shoreline figures to the specified save path.

    Args:
        settings (dict): A dictionary containing the settings for the extraction process.
        save_path (str): The path where the extracted shoreline figures will be saved.
    """
    # Get the data_path and sitename from the settings
    data_path = settings.get("filepath") or settings.get("inputs", {}).get("filepath")
    sitename = settings.get("sitename") or settings.get("inputs", {}).get("sitename")

    # Check if data_path and sitename were successfully retrieved
    if not data_path or not sitename:
        logger.error(f"Data path or sitename not found in settings.{settings}")
        return

    extracted_shoreline_figure_path = os.path.join(
        data_path, sitename, "jpg_files", "detection"
    )
    logger.info(f"extracted_shoreline_figure_path: {extracted_shoreline_figure_path}")

    if os.path.exists(extracted_shoreline_figure_path):
        dst_path = os.path.join(save_path, "jpg_files", "detection")
        logger.info(f"Moving extracted shoreline figures to : {dst_path }")
        file_utilities.move_files(
            extracted_shoreline_figure_path, dst_path, delete_src=True
        )


def convert_linestrings_to_multipoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert LineString geometries in a GeoDataFrame to MultiPoint geometries.

    Args:
    - gdf (gpd.GeoDataFrame): The input GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: A new GeoDataFrame with MultiPoint geometries. If the input GeoDataFrame
                        already contains MultiPoints, the original GeoDataFrame is returned.
    """

    # Check if all geometries in the gdf are MultiPoints
    if all(gdf.geometry.type == "MultiPoint"):
        return gdf

    def linestring_to_multipoint(linestring):
        if isinstance(linestring, LineString):
            return MultiPoint(linestring.coords)
        return linestring

    # Convert each LineString to a MultiPoint
    gdf["geometry"] = gdf["geometry"].apply(linestring_to_multipoint)

    return gdf


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

    # Save extracted shorelines to GeoJSON files
    extracted_shorelines.to_file(
        save_path, "extracted_shorelines_lines.geojson", extracted_shorelines_gdf_lines
    )

    points_gdf = convert_linestrings_to_multipoints(extracted_shorelines.gdf)
    projected_gdf = stringify_datetime_columns(points_gdf)
    # Save extracted shorelines as a GeoJSON file
    extracted_shorelines.to_file(
        save_path, "extracted_shorelines_points.geojson", projected_gdf
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


def create_json_config(
    inputs: dict, settings: dict = {}, roi_ids: list[str] = []
) -> dict:
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
    """Returns true if rois were downloaded before. False if they have not.
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
            "include_T2": False,
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
