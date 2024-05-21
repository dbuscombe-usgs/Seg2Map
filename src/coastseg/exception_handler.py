# standard python imports
import logging
import traceback
from typing import Union

# internal python imports

from coastseg import exceptions
from coastseg import common
from coastseg.roi import ROI

import geopandas as gpd


logger = logging.getLogger(__name__)

SETTINGS = "settings"
SESSION_NAME = "session name"


NO_CONFIG_ROIS = (
    "No ROIs were selected. Cannot save ROIs to config until ROIs are selected."
)
NO_CONFIG_SETTINGS = "Settings must be loaded before configuration files can be made.\nClick save settings"
ROIS_NOT_DOWNLOADED = (
    "Not all ROI directories exist on your computer. Try downloading the ROIs again."
)
BBOX_NOT_FOUND = "Bounding Box not found on map. Please draw a bounding box on the map first."
SESSION_NAME_NOT_FOUND = "No session name found.Enter a session name."
EMPTY_SELECTED_ROIS = (
    "Must select at least one ROI on the map to perform this operation"
)
SETTINGS_NOT_FOUND = "No settings found. Click save settings."
ROI_IS_NONE= "NO ROI found. Try generating a new ROI or loading an ROI from a file."
SHORELINE_NOT_FOUND = "No shoreline found. Please load a shoreline on the map first."
TRANSECTS_NOT_FOUND = "No transects found. Please load a transects on the map first."
NO_ROI_SETTINGS = ("None of the ROIs have been downloaded on this machine or the location where they were downloaded has been moved. Please download the ROIs again.")
NO_EXTRACTED_SHORELINES = "No shorelines have been extracted. Extract shorelines first."
NO_ROIS_WITH_EXTRACTED_SHORELINES = (
    "You must select an ROI and extract shorelines before you can compute transects"
)
NO_CROSS_DISTANCE_TRANSECTS = "No cross distances transects have been computed"

# Separate different exception checking and handling
# Checking decides the message
# Handling it sends message to user

def check_if_default_feature_available(feature, feature_type: str = ""):
    DEFAULT_FEATURE_NOT_FOUND = f"No {feature_type} were available in this region. Draw a new bounding box elsewhere or load a {feature_type} from a file.",

    if isinstance(feature, gpd.GeoDataFrame):
        if feature.empty:
            logger.error(f"{feature_type} is empty {DEFAULT_FEATURE_NOT_FOUND}")
            raise exceptions.Object_Not_Found(feature_type, DEFAULT_FEATURE_NOT_FOUND)
    elif feature is None:
        logger.error(f"{feature_type} is None {DEFAULT_FEATURE_NOT_FOUND}")
        raise exceptions.Object_Not_Found(feature_type, DEFAULT_FEATURE_NOT_FOUND)

def config_check_if_none(feature, feature_type: str = ""):
    """
    Check if a given feature is None and raise an exception if it is.

    Args:
    feature: The feature to be checked.
    feature_type (str): A string representing the type of the feature.

    Raises:
    Object_Not_Found: If the feature is None, an exception is raised with a custom message.
    """
    if feature is None:
        message = (
            f"{feature_type} must be loaded before configuration files can be saved."
        )
        raise exceptions.Object_Not_Found(feature_type, message)


def check_file_not_found(path: str, filename: str, search_path: str):
    """
    Check if a file at a given path is not found and raise a FileNotFoundError if it is.

    Args:
    path (str): The path to be checked.
    filename (str): The name of the file to be checked.
    search_path (str): The search path where the file is expected to be found.

    Raises:
    FileNotFoundError: If the file is not found at the given path.
    """
    # if path is None raises FileNotFoundError
    if path is None:
        logger.error(f"{filename} file was not found at {search_path}")
        raise FileNotFoundError(f"{filename} file was not found at {search_path}")



def check_if_subset(subset: set, superset: set, superset_name: str, message: str = ""):
    """
    Check if a subset is actually a subset of a superset and raise a ValueError if it's not.

    Args:
    subset (set): The subset to be checked.
    superset (set): The superset to be checked against.
    superset_name (str): The name of the superset.
    message (str, optional): Additional message to be included in the error.

    Raises:
    ValueError: If the subset is not a subset of the superset.
    """
    if not subset.issubset(superset):
        logger.error(f"Missing keys {subset-superset} from {superset_name}\n{message}")
        raise ValueError(
            f"Missing keys {subset-superset} from {superset_name}\n{message}</br>Try clicking save settings"
        )


# this function does not do what it claims to do
# def check_if_rois_downloaded(roi_settings: dict, roi_ids: list,data_path:str="/data"):
#     """
#     Check if all ROIs have been downloaded based on the ROI settings and IDs.

#     Args:
#     roi_settings (dict): The settings for the ROIs.
#     roi_ids (list): The list of ROI IDs.

#     Raises:
#     FileNotFoundError: If not all ROIs have been downloaded.
#     """
#     missing_dirs = {}
#     # Check if each ROI selected has been downloaded to the location specified by the config.json (roi settings are derived from this)
#     for roi_id in roi_ids:
#         # check if roi_id directory exists at location specified by filepath in roi_settings
#         if common.were_rois_downloaded(roi_settings, roi_ids) == False:
#             logger.error(f"{roi_id} directory does not exist")
#             missing_dirs[roi_id] = roi_settings[roi_id]["sitename"]
#     # if any of the ROIs were not found in the data dir then raise an exception
#     check_if_dirs_missing(missing_dirs,data_path) 


def can_feature_save_to_file(feature, feature_type: str = ""):
    """
    Check if a feature can be saved to a file, and raise a ValueError if it cannot.

    Args:
    feature: The feature to be checked.
    feature_type (str): A string representing the type of the feature.

    Raises:
    ValueError: If the feature does not exist and thus cannot be saved.
    """
    if feature is None:
        logger.error(f"Feature {feature_type} did not exist. Cannot Save to File")
        raise ValueError(f"Feature {feature_type} did not exist. Cannot Save to File")


def check_empty_dict(feature, feature_type: str = ""):
    """
    Check if a given feature dictionary is empty and raise an appropriate exception.

    Args:
    feature: The feature dictionary to be checked.
    feature_type (str): A string representing the type of the feature.

    Raises:
    Exception: Specific exception based on the feature type if the dictionary is empty.
    """
    if feature == {}:
        if feature_type == "roi_settings":
            raise Exception(NO_ROI_SETTINGS)
        if feature_type == "extracted_shorelines":
            raise Exception(NO_EXTRACTED_SHORELINES)
        if feature_type == "cross_distance_transects":
            raise Exception(NO_CROSS_DISTANCE_TRANSECTS)


def check_empty_layer(layer, feature_type: str = ""):
    """
    Check if a given layer is empty and raise an exception if it is.

    Args:
    layer: The layer to be checked.
    feature_type (str): A string representing the type of the layer.

    Raises:
    Exception: If the layer is empty or None.
    """
    if layer is None:
        if feature_type == ROI.LAYER_NAME:
            logger.error(f"No ROI layer found on map")
            raise Exception("No ROI layer found on map")
        if feature_type == ROI.SELECTED_LAYER_NAME:
            logger.error(f"No selected ROI layer found on map")
            raise Exception("No selected ROI layer found on map")
        logger.error(f"Cannot add an empty {feature_type} layer to the map.")
        raise Exception(f"Cannot add an empty {feature_type} layer to the map.")


def check_if_list_empty(items: list):
    """
    Check if a given list is empty and raise an exception if it is.

    Args:
    items (list): The list to be checked.

    Raises:
    Exception: If the list is empty.
    """
    if len(items) == 0:
        logger.error(f"{items}\n{NO_ROIS_WITH_EXTRACTED_SHORELINES}")
        raise Exception(NO_ROIS_WITH_EXTRACTED_SHORELINES)


def check_if_empty_string(feature, feature_type: str = "", message: str = ""):
    """
    Check if a given feature string is empty and raise an exception if it is.

    Args:
    feature: The feature string to be checked.
    feature_type (str): A string representing the type of the feature.
    message (str, optional): Custom message to be used in the exception.

    Raises:
    Exception: If the feature string is empty.
    """
    if feature == "":
        if feature_type == SESSION_NAME:
            message = SESSION_NAME_NOT_FOUND
        logger.error(f"{feature_type} is empty string")
        raise Exception(message)


def check_if_None(feature, feature_type: str = "", message: str = ""):
    """
    Check if a given feature is None and raise an exception if it is.

    Args:
    feature: The feature to be checked.
    feature_type (str): A string representing the type of the feature.
    message (str, optional): Custom message to be used in the exception.

    Raises:
    Object_Not_Found: If the feature is None.
    """
    if feature is None:
        if feature_type == "settings":
            message = SETTINGS_NOT_FOUND
        if feature_type == "shoreline":
            message = SHORELINE_NOT_FOUND
        if "roi" in feature_type.lower():
            message = ROI_IS_NONE
        logger.error(f"{feature_type} is None")
        raise exceptions.Object_Not_Found(feature_type, message)

def validate_feature(feature, feature_type: str = "", message: str = ""):
    """
    Check if a given feature is None and raise an exception if it is.

    Args:
    feature: The feature to be checked.
    feature_type (str): A string representing the type of the feature.
    message (str, optional): Custom message to be used in the exception.

    Raises:
    Object_Not_Found: If the feature is None, an exception is raised with a custom message.
    """
    if feature is None and not hasattr(feature,"gdf"):
        if "shoreline" in feature_type.lower():
            message = SHORELINE_NOT_FOUND
        if "roi" in feature_type.lower():
            message = ROI_IS_NONE
        if "transect" in feature_type.lower():
            message = TRANSECTS_NOT_FOUND
        if "bbox" in feature_type.lower():
            message = BBOX_NOT_FOUND
               
        logger.error(f"{feature_type} was not found. {message}")
        raise exceptions.Object_Not_Found(feature_type, message)


def check_empty_roi_layer(layer):
    """
    Check if the ROI layer is empty and raise an exception if it is.

    Args:
    layer: The ROI layer to be checked.

    Raises:
    Exception: If the ROI layer is empty.
    """
    if layer is None:
        raise Exception(EMPTY_SELECTED_ROIS)


def check_selected_set(selected_set:set):
    """
    Check if the selected set is empty and raise an exception if it is.

    Args:
    selected_set(set): The set to be checked.

    Raises:
    Exception: If the selected set is empty.
    """
    if selected_set is None:
        raise Exception(EMPTY_SELECTED_ROIS)
    if len(selected_set) == 0:
        raise Exception(EMPTY_SELECTED_ROIS)


def check_if_gdf_empty(feature, feature_type: str, message: str = ""):
    """
    Check if a given GeoDataFrame is empty and raise an exception if it is.

    Args:
    feature: The GeoDataFrame to be checked.
    feature_type (str): A string representing the type of the feature.
    message (str, optional): Custom message to be used in the exception.

    Raises:
    Object_Not_Found: If the GeoDataFrame is empty.
    """
    if feature.empty == True:
        logger.error(f"{feature_type} {feature} is empty")
        raise exceptions.Object_Not_Found(feature_type, message)


def check_if_dirs_missing(missing_dirs: dict,location:str="/data"):
    """
    Check if there are any missing directories and raise a WarningException if there are.

    Args:
    missing_dirs (dict):The dictionary of each ROI and the name of the of missing directory
    example: {roi_id1: 'missing_dir_name'}
    location (str, optional): The location where the directories are missing.
    message (str, optional): Custom message to be used in the exception.

    Raises:
    WarningException: If there are missing directories.
    """
    if len(missing_dirs) != 0:
        if not location:
            location = "/data"
        # format the missing directories into a string
        # missing_dirs_str = "</br>".join([f"<span style='color: red'><i><b>{roi_id}</b> was missing the folder '<u>{folder}</u>'</i></span>" for roi_id, folder in missing_dirs.items()])

        logger.error(
            f"The following ROIs that were in the config.json file are missing their data:</br> \n {missing_dirs}"
        )
        raise exceptions.WarningMissingDirsException(
            message=f"The following ROIs that were in the config.json file are missing their data:", 
            instructions=f"You can't extract shorelines for the ROIs missing data, but you can for the rest of your ROIs.\nMake sure to de-select these ROIs before extracting shorelines.\n Before you can extract shorelines the missing ROIs you can either:\n 1. Move the missing directories into '{location}' and reload the session \n 2. Download the missing data in a new session ",
            styled_instructions=f"You can't extract shorelines for the ROIs missing data, but you can for the rest of your ROIs.</br>Make sure to de-select these ROIs before extracting shorelines.</br> </br> Before you can extract shorelines the missing ROIs you can either:</br> 1. Move the missing directories into <u>'{location}'</u> and reload the session </br> 2. Download the missing data in a new session ",
            missing_dirs=missing_dirs
        )   


def handle_exception(error:Exception, row: "ipywidgets.HBox", title: str = None, msg: str = None):
    """
    Handle exceptions by logging and displaying them in a user interface.

    Args:
    error(Exception): The exception to be handled.
    row (ipywidgets.HBox): The UI row where the error message will be displayed.
    title (str, optional): The title for the error message.
    msg (str, optional): The custom message for the error.

    Returns:
    None
    """

    logger.error(f"{traceback.format_exc()}")
    if isinstance(error, exceptions.WarningMissingDirsException):
        logger.error(f"error.instructions: {error.instructions}")
        launch_error_box(
            row,
            title="Warning Data Missing",
            msg=error.get_styled_message(),
            instructions=error.get_instructions(),
        )
    elif isinstance(error, exceptions.WarningException):
        logger.error(f"error.instructions: {error.instructions}")
        launch_error_box(
            row,
            title="Warning Data Missing",
            msg=error.msg,
            instructions=error.instructions,
        )
    else:
        error_message = (
            f"{error}</br>Additional Information</br>" + traceback.format_exc()
        )
        if isinstance(error, exceptions.Object_Not_Found):
            error_message = str(error)
        logger.error(f"{error_message}")
        launch_error_box(row, title="Error", msg=error_message)


# def handle_warning(warning, row: "ipywidgets.HBox", title: str = None, msg: str = None):
#     error_message = f"{warning}"
#     logger.error(f"{traceback.format_exc()}")
#     if isinstance(warning, exceptions.Object_Not_Found):
#         error_message = str(warning)
#     logger.error(f"{error_message}")
#     launch_error_box(row, title="Error", msg=error_message)


def handle_bbox_error(error_msg: Union[exceptions.BboxTooLargeError, exceptions.BboxTooSmallError], row: "ipywidgets.HBox"):
    """
    Handle bounding box related errors.

    Args:
    error_msg (Union[exceptions.BboxTooLargeError, exceptions.BboxTooSmallError]): The error message related to bounding box issues.
    row (ipywidgets.HBox): The UI row where the error message will be displayed.

    Returns:
    None
    """
    logger.error(f"Bounding Box Error{error_msg}")
    launch_error_box(row, title="Error", msg=error_msg)


def launch_error_box(row: "ipywidgets.HBox", title: str = None, msg: str = None, instructions: str = None):
    """
    Launch an error box in the user interface to display error messages.

    Args:
    row (ipywidgets.HBox): The UI row where the error box will be displayed.
    title (str, optional): The title for the error box.
    msg (str, optional): The message to be displayed in the error box.
    instructions (str, optional): Additional instructions to be displayed in the error box.
    """
    # Show user error message
    warning_box = common.create_warning_box(
        title=title,
        msg=msg,
        instructions=instructions,
        msg_width="100%",
        box_width="80%",
    )
    # clear row and close all widgets in self.file_row before adding new warning_box
    common.clear_row(row)
    # add instance of warning_box to row
    row.children = [warning_box]
