# Standard library imports
import os
import math
import json
import logging
import glob
from datetime import datetime
from collections import defaultdict
from typing import Collection, Dict, List, Optional, Tuple, Union
import traceback

# Third-party imports
import geopandas as gpd
import pandas as pd
from ipyleaflet import DrawControl, LayersControl, WidgetControl, GeoJSON
from leafmap import Map
from ipywidgets import Layout, HTML, HBox
from tqdm.auto import tqdm
import traitlets

# Internal/Local imports: specific classes/functions
from coastseg.bbox import Bounding_Box
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI
from coastseg.downloads import count_images_in_ee_collection
from coastseg import file_utilities
from coastseg import geodata_processing
from coastseg import tide_correction

# Internal/Local imports: modules
from coastseg import (
    common,
    factory,
    exceptions,
    extracted_shoreline,
    exception_handler,
)
from coastsat import SDS_download
from coastsat.SDS_download import get_metadata

logger = logging.getLogger(__name__)

SELECTED_LAYER_NAME = "Selected Shorelines"

__all__ = ["IDContainer", "ExtractShorelinesContainer", "CoastSeg_Map"]


class IDContainer(traitlets.HasTraits):
    ids = traitlets.List(trait=traitlets.Unicode())


class ExtractShorelinesContainer(traitlets.HasTraits):
    """A container class for managing shorelines extraction.

    This class provides a container for managing shorelines extraction.
    It holds lists of shorelines that can be loaded, shorelines that will be thrown away, and
    ROI (Region of Interest) IDs that have extracted shorelines.

    Args:
        traitlets (type): The traitlets module for defining traits.

    Attributes:
        load_list (List[str]): A list of shorelines that can be loaded.
        trash_list (List[str]): A list of shorelines that will be thrown away.
        roi_ids_list (List[str]): A list of ROI IDs that have extracted shorelines.
    """

    # list of shorelines that can be loaded
    load_list = traitlets.List(trait=traitlets.Unicode())
    # list of shorelines that will be thrown away
    trash_list = traitlets.List(trait=traitlets.Unicode())
    # list of roi ids that have extracted shorelines
    roi_ids_list = traitlets.List(trait=traitlets.Unicode())

    def __init__(
        self, load_list_widget=None, trash_list_widget=None, roi_list_widget=None
    ):
        super().__init__()
        if load_list_widget:
            self.link_load_list(load_list_widget)
        if trash_list_widget:
            self.link_trash_list(trash_list_widget)
        # Link the widgets and the traits
        if roi_list_widget:
            self.link_roi_list(roi_list_widget)

    def link_load_list(self, widget):
        if hasattr(widget, "options"):
            traitlets.dlink((self, "load_list"), (widget, "options"))

    def link_trash_list(self, widget):
        if hasattr(widget, "options"):
            traitlets.dlink((self, "trash_list"), (widget, "options"))

    def link_roi_list(self, widget):
        if hasattr(widget, "options"):
            traitlets.dlink((self, "roi_ids_list"), (widget, "options"))

    def clear(self):
        self.load_list = []
        self.trash_list = []
        self.roi_ids_list = []


def find_shorelines_directory(path, roi_id):
    # List the contents of the specified path
    contents = os.listdir(path)

    # Check for extracted shorelines geojson file in the specified path
    extracted_shorelines_file = [
        file
        for file in contents
        if "extracted_shorelines" in file and file.endswith(".geojson")
    ]
    if extracted_shorelines_file:
        return path

    # If the file is not found, check for a directory with the ROI ID
    roi_directory = [
        directory
        for directory in contents
        if os.path.isdir(os.path.join(path, directory)) and roi_id in directory
    ]
    if roi_directory:
        roi_path = os.path.join(path, roi_directory[0])
        roi_contents = os.listdir(roi_path)
        extracted_shorelines_file = [
            file
            for file in roi_contents
            if "extracted_shorelines" in file and file.endswith(".geojson")
        ]
        if extracted_shorelines_file:
            return roi_path

    return None


def find_shorelines_directory(path, roi_id):
    # List the contents of the specified path
    contents = os.listdir(path)

    # Check for extracted shorelines geojson file in the specified path
    extracted_shorelines_file = [
        file
        for file in contents
        if "extracted_shorelines" in file and file.endswith(".geojson")
    ]
    if extracted_shorelines_file:
        return path

    # If the file is not found, check for a directory with the ROI ID
    roi_directory = [
        directory
        for directory in contents
        if os.path.isdir(os.path.join(path, directory)) and roi_id in directory
    ]
    if roi_directory:
        roi_path = os.path.join(path, roi_directory[0])
        roi_contents = os.listdir(roi_path)
        extracted_shorelines_file = [
            file
            for file in roi_contents
            if "extracted_shorelines" in file and file.endswith(".geojson")
        ]
        if extracted_shorelines_file:
            return roi_path

    return None


def delete_extracted_shorelines_files(session_path: str, selected_items: List):
    """
    Delete the extracted shorelines from the session directory for all the relevant files.

    Args:
        session_path (str): The path to the session directory.
        selected_items: A list of strings, where each string is in the format "satname_dates".
        This is a string that represents the satellite name and the dates of the extracted shoreline.

    Removes the extracted shorelines from the following files:
    - extracted_shorelines_lines.geojson
    - extracted_shorelines_points.geojson
    - extracted_shorelines_dict.json
    - transect_time_series.csv
    - transect_time_series_tidally_corrected.csv

    As well as all the csv files matching the following patterns:
    - _timeseries_raw.csv
    - _timeseries_tidally_corrected.csv
    """
    # Extract dates and satellite names from the selected items
    dates_list, sat_list = common.extract_dates_and_sats(selected_items)

    # formatted_dates = [datetime.datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S") for datestr in dates_list],
    # delete the extracted shorelines from both geojson files
    filenames = [
        "extracted_shorelines_lines.geojson",
        "extracted_shorelines_points.geojson",
    ]
    filepaths = [
        os.path.join(session_path, filename)
        for filename in filenames
        if os.path.isfile(os.path.join(session_path, filename))
    ]
    # the date column must be a dateime formatted in "%Y-%m-%d %H:%M:%S" without a timezone
    formatted_dates = [date.replace(tzinfo=None) for date in dates_list]
    # edit the shoreline geojson files and remove the selected shorelines that having matching date and satname
    geodata_processing.edit_geojson_files(
        filepaths, common.remove_matching_rows, date=formatted_dates, satname=sat_list
    )
    # delete the extracted shorelines from the extracted_shorelines_dict.json file
    filename = "extracted_shorelines_dict.json"
    # update extracted_shorelines_dict.json and transects_cross_distances.json
    common.update_extracted_shorelines_dict_transects_dict(
        session_path, filename, dates_list, sat_list
    )
    # delete the extracted shorelines from the transect_time_series.csv files
    filenames = [
        "transect_time_series.csv",
        "transect_time_series_tidally_corrected.csv",
    ]
    filepaths = [
        os.path.join(session_path, filename)
        for filename in filenames
        if os.path.isfile(os.path.join(session_path, filename))
    ]
    common.update_transect_time_series(filepaths, dates_list)
    # delete the selected shorelines from all the individual csv files
    file_patterns = ["_timeseries_tidally_corrected", "_timeseries_raw.csv"]
    for file_pattern in file_patterns:
        common.drop_dates_from_csv(file_pattern, session_path, dates_list)
    # delete the extracted shorelines from the jpg detection files
    jpg_path = os.path.join(session_path, "jpg_files", "detection")
    if os.path.exists(jpg_path) and os.path.isdir(jpg_path):
        common.delete_jpg_files(dates_list, sat_list, jpg_path)


class CoastSeg_Map:
    def __init__(self):
        # Basic settings and configurations
        self.settings = {}
        self.set_settings()
        self.session_name = ""

        # Factory for creating map objects
        self.factory = factory.Factory()

        # Observables
        self.id_container = IDContainer(ids=[])
        self.extract_shorelines_container = ExtractShorelinesContainer()

        # Map objects and configurations
        self.rois = None
        self.transects = None
        self.shoreline = None
        self.bbox = None
        self.selected_set = set()
        self.selected_shorelines_set = set()
        self._init_map_components()

        # Warning and information boxes
        self._init_info_boxes()

    def _init_map_components(self):
        """Initialize map-related attributes and settings."""
        self.map = self.create_map()
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.map.add(self.draw_control)
        self.map.add(LayersControl(position="topright"))

    def _init_info_boxes(self):
        """Initialize info and warning boxes for the map."""
        self.warning_box = HBox([],layout=Layout(height='242px'))
        self.warning_widget = WidgetControl(widget=self.warning_box, position="topleft")
        self.map.add(self.warning_widget)

        self.roi_html = HTML("""""")
        self.roi_box = common.create_hover_box(title="ROI", feature_html=self.roi_html)
        self.roi_widget = WidgetControl(widget=self.roi_box, position="topright")
        self.map.add(self.roi_widget)

        self.feature_html = HTML("""""")
        self.hover_box = common.create_hover_box(
            title="Feature", feature_html=self.feature_html
        )
        self.hover_widget = WidgetControl(widget=self.hover_box, position="topright")
        self.map.add(self.hover_widget)

    def __str__(self):
        return f"CoastSeg: roi={self.rois}\n shoreline={self.shoreline}\n  transects={self.transects}\n bbox={self.bbox}"

    def __repr__(self):
        return f"CoastSeg(roi={self.rois}\n shoreline={self.shoreline}\n transects={self.transects}\n bbox={self.bbox} "

    def get_session_name(self):
        return self.session_name

    def set_session_name(self, name: str):
        self.session_name = name

    def load_extracted_shoreline_layer(self, gdf, layer_name, colormap):
        map_crs = "epsg:4326"
        # create a layer with the extracted shorelines selected
        points_gdf = extracted_shoreline.convert_linestrings_to_multipoints(gdf)
        projected_gdf = points_gdf.to_crs(map_crs)

        import matplotlib.pyplot as plt
        import numpy as np

        # convert date column to datetime
        projected_gdf["date"] = pd.to_datetime(projected_gdf["date"])
        # Sort the GeoDataFrame based on the 'date' column
        projected_gdf = projected_gdf.sort_values(by="date")
        # normalize the dates to 0-1 scale
        min_date = projected_gdf["date"].min()
        max_date = projected_gdf["date"].max()
        if min_date == max_date:
            # If there's only one date, set delta to 0.25
            delta = np.array([0.25])
        else:
            delta = (projected_gdf["date"] - min_date) / (max_date - min_date)
        # get the colors from the colormap
        colors = plt.cm.get_cmap(colormap)(delta)

        # convert RGBA colors to Hex
        colors_hex = [
            "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, a in colors
        ]

        # add the colors to the GeoDataFrame
        projected_gdf["color"] = colors_hex

        projected_gdf = common.stringify_datetime_columns(projected_gdf)
        # Convert GeoDataFrame to GeoJSON
        features_json = json.loads(projected_gdf.to_json())

        # Add 'id' field to features in GeoJSON
        for feature, index in zip(
            features_json["features"], range(len(features_json["features"]))
        ):
            feature["id"] = str(index)

        # define a style callback function that takes a feature and returns a style dictionary
        def style_callback(feature):
            # find the color for the current feature based on its id
            color = projected_gdf.iloc[int(feature["id"])]["color"]
            return {"color": color, "weight": 5, "fillColor": color, "fillOpacity": 0.5}

        # create an ipyleaflet GeoJSON layer with the extracted shorelines selected
        new_layer = GeoJSON(
            data=features_json,
            name=layer_name,
            style_callback=style_callback,
            point_style={
                "radius": 1,
                "opacity": 1,
            },
        )
        # features_json = json.loads(projected_gdf.to_json())
        # # create an ipyleaflet GeoJSON layer with the extracted shorelines selected
        # new_layer = GeoJSON(
        #     data=features_json, name=layer_name, style=style, point_style=style
        # )
        self.replace_layer_by_name(layer_name, new_layer, on_hover=None, on_click=None)

    def delete_selected_shorelines(
        self, layer_name: str, selected_id: str, selected_shorelines: List = None
    ) -> None:
        if selected_shorelines and selected_id:
            self.map.default_style = {"cursor": "wait"}
            session_name = self.get_session_name()
            session_path = file_utilities.get_session_location(
                session_name=session_name, raise_error=True
            )
            # get the path to the session directory that contains the extracted shoreline files
            session_path = find_shorelines_directory(session_path, selected_id)
            # remove the extracted shorelines from the extracted shorelines object stored in the ROI
            dates, satellites = common.extract_dates_and_sats(selected_shorelines)
            self.rois.remove_selected_shorelines(selected_id, dates, satellites)
            # remove_selected_shorelines
            # remove the extracted shorelines from the files in the session location
            if os.path.exists(session_path) and os.path.isdir(session_path):
                delete_extracted_shorelines_files(
                    session_path, list(selected_shorelines)
                )
            # this will remove the selected shorelines from the files
        self.remove_layer_by_name(layer_name)
        self.map.default_style = {"cursor": "default"}

    def load_selected_shorelines_on_map(
        self,
        selected_id: str,
        selected_shorelines: List,
        layer_name: str,
        colormap: str,
    ) -> None:
        def get_selected_shorelines(gdf, selected_items: list[str]) -> gpd.GeoDataFrame:
            """
            Filter the GeoDataFrame based on selected items.

            Args:
                gdf (gpd.GeoDataFrame): The input GeoDataFrame.
                selected_items (list[str]): A list of selected items in the format "satname_dates".

            Returns:
                gpd.GeoDataFrame: The filtered GeoDataFrame containing the selected shorelines.
            """
            # Filtering criteria
            frames = []  # List to collect filtered frames
            # Loop through each dictionary in dates_tuple
            for criteria in list(selected_items):
                satname, dates = criteria.split("_")
                # if "date" column string already then don't convert to datetime
                if gdf["date"].dtype == "object":
                    filtered = gdf[(gdf["date"] == dates) & (gdf["satname"] == satname)]
                else:
                    filtered = gdf[
                        (gdf["date"] == datetime.strptime(dates, "%Y-%m-%d %H:%M:%S"))
                        & (gdf["satname"] == satname)
                    ]
                frames.append(filtered)

            # Concatenate the frames to get the final result
            filtered_gdf = gpd.GeoDataFrame(columns=["geometry"])
            filtered_gdf.crs = "epsg:4326"
            if frames:
                filtered_gdf = pd.concat(frames)
            return filtered_gdf

        # load the extracted shorelines for the selected ROI ID
        if self.rois is not None:
            extracted_shorelines = self.rois.get_extracted_shoreline(selected_id)
            # get the geodataframe for the extracted shorelines
            if hasattr(extracted_shorelines, "gdf"):
                selected_gdf = get_selected_shorelines(
                    extracted_shorelines.gdf, selected_shorelines
                )
                logger.info(
                    f"load_selected_shorelines_on_map: selected_gdf.head() {selected_gdf.head()}"
                )
                if not selected_gdf.empty:
                    self.load_extracted_shoreline_layer(
                        selected_gdf, layer_name, colormap
                    )

    def on_roi_change(
        self,
        selected_id: str,
    ) -> None:
        # remove the old layers
        self.remove_extracted_shoreline_layers()
        # update the load_list and trash_list
        extracted_shorelines = self.update_loadable_shorelines(selected_id)
        self.extract_shorelines_container.trash_list = []
        # load the new extracted shorelines onto the map
        self.load_extracted_shorelines_on_map(extracted_shorelines, 0)

    def create_map(self):
        """create an interactive map object using the map_settings
        Returns:
           ipyleaflet.Map: ipyleaflet interactive Map object
        """
        map_settings = {
            "center_point": (36.8470, -121.8024),
            "zoom": 7,
            "draw_control": False,
            "measure_control": False,
            "fullscreen_control": False,
            "attribution_control": True,
            "Layout": Layout(width="100%", height="100px"),
        }
        return Map(
            draw_control=map_settings["draw_control"],
            measure_control=map_settings["measure_control"],
            fullscreen_control=map_settings["fullscreen_control"],
            attribution_control=map_settings["attribution_control"],
            center=map_settings["center_point"],
            zoom=map_settings["zoom"],
            layout=map_settings["Layout"],
            world_copy_jump=True,
        )

    def compute_tidal_corrections(
        self, roi_ids: Collection, beach_slope: float, reference_elevation: float
    ):
        logger.info(
            f"Computing tides for ROIs {roi_ids} beach_slope: {beach_slope} reference_elevation: {reference_elevation}"
        )

        session_name = self.get_session_name()
        try:
            tide_correction.correct_all_tides(
                roi_ids,
                session_name,
                reference_elevation,
                beach_slope,
            )
        except Exception as e:
            exception_handler.handle_exception(
                e,
                self.warning_box,
                title="Tide Model Error",
                msg=str(e),
            )
        else:
            print("\ntidal corrections completed")

    def load_metadata(self, settings: dict = {}, ids: Collection = set([])):
        """
        Loads metadata either based on user-provided settings or a collection of ROI IDs.
        This also creates a metadata file for each ROI in the data directory.

        This method either takes in a dictionary with site-specific settings to load metadata
        for a particular site, or iterates over a collection of ROI IDs to load their respective
        metadata using the settings associated with those ROI IDs.

        Note that coastsat's `get_metadata` is used to actually perform the metadata loading.

        Parameters:
        -----------
        settings: dict, optional
            A dictionary containing settings for a specific site. The settings should
            include 'sitename' and 'filepath_data' keys, among others. Default is an empty dict.

        ids: Collection, optional
            A collection (e.g., set, list) of ROI IDs to load metadata for. Default is an empty set.

        Raises:
        -----------
        FileNotFoundError:
            If the directory specified in the settings or by ROI IDs does not exist.

        Exception:
            If neither settings nor ids are provided.

        Returns:
        -----------
        None
            metadata dictionary

        Examples:
        -----------
        >>> load_metadata(settings={'sitename': 'site1', 'filepath_data': '/path/to/data'})
        >>> load_metadata(ids={1, 2, 3})

        """
        if settings and isinstance(settings, dict):
            return get_metadata(settings)
        elif ids:
            for roi_id in ids:
                # if the ROI directory did not exist then print a warning and proceed
                try:
                    logger.info(
                        f"Loading metadata using {self.rois.roi_settings[str(roi_id)]}"
                    )
                    metadata = get_metadata(self.rois.roi_settings[str(roi_id)])
                    logger.info(f"Metadata for ROI ID {str(roi_id)}:{metadata}")
                    return metadata
                except FileNotFoundError as e:
                    logger.error(f"Metadata not loaded for ROI ID {str(roi_id)} {e}")
                    print(f"Metadata not loaded for ROI ID {str(roi_id)} {e}")

        else:
            raise Exception(f"Must provide settings or list of IDs to load metadata.")

    def load_session_files(self, dir_path: str,data_path:str="") -> None:
        """
        Load the configuration files from the given directory.

        The function looks for the following files in the directory:
        - config_gdf.geojson: contains the configuration settings for the project
        - transects_settings.json: contains the settings for the transects module
        - shoreline_settings.json: contains the settings for the shoreline module

        If the config_gdf.geojson file is not found, a message is printed to the console.

        Args:
            dir_path (str): The path to the directory containing the configuration files.
            data_path (str): Full path to the coastseg data directory where downloaded data is saved
        Returns:
            None
        """
        if os.path.isdir(dir_path):
            # ensure coastseg\data location exists
            if not data_path:
                data_path = file_utilities.create_directory(os.getcwd(), "data")
            config_geojson_path = os.path.join(dir_path, "config_gdf.geojson")
            config_json_path = os.path.join(dir_path, "config.json")
            # load the config files if they exist
            config_loaded = self.load_config_files(data_path, config_geojson_path, config_json_path)
            # create metadata files for each ROI loaded in using coastsat's get_metadata()
            if self.rois and getattr(self.rois, "roi_settings"):
                self.load_metadata(ids=list(self.rois.roi_settings.keys()))
            else:
                logger.warning(f"No ROIs were able to have their metadata loaded.")
            # load in setting from shoreline_settings.json and transects_settings.json
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if not os.path.isfile(file_path):
                    continue
                if file_name == "shoreline_settings.json":
                    keys = [
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
                    ]
                    settings = common.load_settings(file_path, keys)
                    self.set_settings(**settings)
                elif file_name == "transects_settings.json":
                    keys = [
                        "max_std",
                        "min_points",
                        "along_dist",
                        "max_range",
                        "min_chainage",
                        "multiple_inter",
                        "prc_multiple",
                    ]
                    settings = common.load_settings(file_path, keys)
                    self.set_settings(**settings)
            if not config_loaded:
                logger.info(f"Not all config files not found at {dir_path}")

    def load_session_from_directory(self, dir_path: str,data_path:str="") -> None:
        """
        Loads a session from a specified directory path.
        Loads config files, extracted shorelines, and transects & extracted shoreline intersections.

        Args:
            dir_path (str): The path of the directory to load the session from.

        Returns:
            None. The function updates the coastseg instance with ROIs, extracted shorelines, and transects
        """
        self.load_session_files(dir_path,data_path)
        # for every directory load extracted shorelines
        extracted_shorelines = extracted_shoreline.load_extracted_shoreline_from_files(
            dir_path
        )
        if extracted_shorelines is None:
            logger.warning(f"No extracted shorelines found in {dir_path}")
            return
        # get roi id from extracted shoreline
        roi_id = extracted_shorelines.get_roi_id()
        if roi_id is None:
            logger.warning(
                f"No roi id found extracted shorelines settings{extracted_shorelines.shoreline_settings}"
            )
            return

        # add extracted shoreline to ROI it was extracted from
        if self.rois is not None:
            self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)
            # load extracted shoreline and transect intersections
            cross_distances = common.load_cross_distances_from_file(dir_path)
            # add extracted shoreline and transect intersections to ROI they were extracted from
            self.rois.add_cross_shore_distances(cross_distances, roi_id)

    def load_fresh_session(self, session_path: str) -> None:
        """
        Load a fresh session by removing all the old features from the map and loading a new session.

        Args:
            session_path (str): The path to the session directory

        Returns:
            None
        """
        # remove all the old features from the map
        self.remove_all()
        self.load_session(session_path)

    def load_session(self, session_path: str,data_path:str="") -> None:
        """
        Load a session from the given path.

        The function loads a session from the given path, which can contain one or more directories, each containing
        the files for a single ROI. For each subdirectory, the function calls `load_session_from_directory` to load
        the session files and objects on the map. If no subdirectories exist, the function calls `load_session_from_directory` with the
        session path.

        Args:
            session_path: The path to the session directory.
            data_path (str): Full path to the coastseg data directory where downloaded data is saved
        Returns:
            None.
        """

        def get_parent_session_name(session_path: str) -> str:
            split_array = session_path.split(os.sep)
            # get the index of the sessions directory which contains all the sessions
            if "data" in split_array:
                return os.path.basename(session_path)
            if "sessions" in split_array:
                parent_index = split_array.index("sessions")
            # get the parent session name aka not a sub directory for a specific ROI
            parent_session_name = split_array[parent_index + 1]
            if not (
                os.path.exists(os.sep.join(split_array[: parent_index + 1]))
                and os.path.isdir(os.sep.join(split_array[: parent_index + 1]))
            ):
                raise FileNotFoundError(f"{os.sep.join(split_array[:parent_index+1])}")
            return parent_session_name

        if not data_path:
            data_path = file_utilities.create_directory(os.getcwd(), "data")

        # load the session name
        session_path = os.path.abspath(session_path)

        session_name = get_parent_session_name(session_path)
        logger.info(f"session_name: {session_name} session_path: {session_path}")
        self.set_session_name(session_name)
        logger.info(f"Loading session from session directory: {session_path}")

        # load the session from the parent directory and subdirectories within session path
        directories_to_load = file_utilities.get_all_subdirectories(session_path)
        for directory in directories_to_load:
            self.load_session_from_directory(directory,data_path)

        # update the list of roi's ids who have extracted shorelines
        ids_with_extracted_shorelines = self.update_roi_ids_with_shorelines()
        logger.info(
            f"Available roi_ids from extracted shorelines: {ids_with_extracted_shorelines}"
        )
         
        # get the ROIs and check if they have settings 
        if self.rois is not None:
            roi_settings = self.rois.get_roi_settings()
            logger.info(f"Checking roi_settings for missing ROIs: {roi_settings}")
            # check if any of the ROIs were missing in in the data directory
            missing_directories = common.get_missing_roi_dirs(roi_settings)
            logger.info(f"Missing directories: {missing_directories}")
            # raise a warning message if any of the ROIs were missing in the data directory
            exception_handler.check_if_dirs_missing(missing_directories,data_path)

    def load_gdf_config(self, filepath: str) -> None:
        """Load features from geodataframe located in geojson file at filepath onto map.

        Features in config file should contain a column named "type" which contains one of the
        following possible feature types: "roi", "shoreline", "transect", "bbox".

        Args:
            filepath (str): full path to config_gdf.geojson
        """

        gdf = geodata_processing.read_gpd_file(filepath)
        gdf = common.stringify_datetime_columns(gdf)

        # each possible type of feature and the columns that should be loaded
        feature_types = {
            "bbox": ["geometry"],
            "roi": ["id", "geometry"],
            "transect": list(Transects.COLUMNS_TO_KEEP),
            "shoreline": ["geometry"],
        }

        for feature_name, columns in feature_types.items():
            feature_gdf = self._extract_feature_gdf(gdf, feature_name, columns)
            if feature_name == "roi":
                exception_handler.check_if_gdf_empty(
                    feature_gdf, "ROIs", "Cannot load empty ROIs onto map"
                )
                if self.rois is None:
                    self.rois = ROI(rois_gdf=feature_gdf)
                    self.load_feature_on_map(
                        feature_name, gdf=feature_gdf, zoom_to_bounds=True
                    )
                elif self.rois is not None:
                    # add the new roi to the existing rois
                    self.rois = self.rois.add_geodataframe(feature_gdf)
                    # load the new rois onto the map
                    self.add_feature_on_map(self.rois, feature_name)
            else:
                # load shorelines, transects, or bbox features onto the map
                self.load_feature_on_map(
                    feature_name, gdf=feature_gdf, zoom_to_bounds=True
                )
        del gdf

    def _extract_feature_gdf(
        self, gdf: gpd.GeoDataFrame, feature_type: str, columns: List[str]
    ) -> gpd.GeoDataFrame:
        """
        Extracts a GeoDataFrame of features of a given type and specified columns from a larger GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): The GeoDataFrame containing the features to extract.
            feature_type (str): The type of feature to extract.
            columns (List[str]): A list of column names to extract from the GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: A new GeoDataFrame containing only the features of the specified type and columns.

        Raises:
            ValueError: Raised when feature_type or any of the columns specified do not exist in the GeoDataFrame.
        """
        # Check if feature_type exists in the GeoDataFrame
        if "type" not in gdf.columns:
            raise ValueError(
                f"Column 'type' does not exist in the GeoDataFrame. Incorrect config_gdf.geojson loaded"
            )

        # select only the columns that are in the gdf
        keep_columns = [col for col in columns if col in gdf.columns]

        # If no columns from columns list exist in the GeoDataFrame, raise an error
        if not keep_columns:
            raise ValueError(
                f"None of the columns {columns} exist in the GeoDataFrame."
            )

        # select only the features that are of the correct type and have the correct columns
        feature_gdf = gdf[gdf["type"] == feature_type][keep_columns]

        return feature_gdf

    def preview_available_images(self):
        """
        Preview the available satellite images for selected regions of interest (ROIs).

        This function checks if ROIs exist and if one has been selected. It then retrieves
        the start and end dates from the settings and iterates over each selected ROI ID.
        For each ROI, it extracts the polygonal geometry, queries the satellite image collections
        using `count_images_in_ee_collection`, and prints the count of available images for each
        satellite.

        It provides a progress bar using `tqdm` to indicate the processing of each ROI.

        Attributes:
        rois (object): An object that should contain the ROIs, including a GeoDataFrame (`gdf` attribute)
                    with "id" and "geometry" columns.
        selected_set (iterable): A set or list of ROI IDs that have been selected for processing.
        settings (dict): A dictionary containing configuration settings, including "dates" which is
                        a list containing the start and end date in the format ['YYYY-MM-DD', 'YYYY-MM-DD'].

        Raises:
        Exception: If no ROIs are provided, if the ROIs GeoDataFrame is empty, or if no ROI has been selected.

        Prints:
        The ROI ID and the count of available images for each satellite.

        Example usage:
        >>> self.rois = <...>  # Load ROIs into the object
        >>> self.selected_set = {1, 2, 3}  # Example IDs of selected ROIs
        >>> self.settings = {"dates": ['2022-01-01', '2022-12-31']}
        >>> preview_available_images()
        ROI ID: 1
        L5: 10 images
        L7: 8 images
        L8: 12 images
        L9: 5 images
        S2: 11 images
        """
        # check that ROIs exist and one has been clicked
        exception_handler.check_if_None(self.rois, "ROI")
        exception_handler.check_if_gdf_empty(self.rois.gdf, "ROI")
        exception_handler.check_selected_set(self.selected_set)
        # get the start and end date to check available images
        start_date, end_date = self.settings["dates"]
        # for each selected ID return the images available for each site
        for roi_id in tqdm(self.selected_set, desc="Processing", leave=False):
            polygon = common.get_roi_polygon(self.rois.gdf, roi_id)
            if polygon:
                # only get the imagery in tier 1
                images_count = count_images_in_ee_collection(
                    polygon,
                    start_date,
                    end_date,
                    satellites=set(self.settings["sat_list"]),
                    tiers=[1],
                )
                satellite_messages = [f"\nROI ID: {roi_id}"]
                for sat in self.settings["sat_list"]:
                    satellite_messages.append(f"{sat}: {images_count[sat]} images")

                print("\n".join(satellite_messages))

    def download_imagery(self) -> None:
        """
        Downloads all images for the selected ROIs  from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2  covering the area of interest and acquired between the specified dates.
        The downloaded imagery for each ROI is stored in a directory that follows the convention
        ID_{ROI}_datetime{current date}__{time}' ex.ID_0_datetime04-11-23__10_20_48. The files are saved as jpgs in a subdirectory
        'jpg_files' in a subdirectory 'preprocessed' which contains subdirectories for RGB, NIR, and SWIR jpg imagery. The downloaded .TIF images are organised in subfolders, divided
        by satellite mission. The bands are also subdivided by pixel resolution.

        Raises:
            Exception: raised if settings is missing
            Exception: raised if 'dates','sat_list', and 'landsat_collection' are not in settings
            Exception: raised if no ROIs have been selected
        """

        self.validate_download_imagery_inputs()

        # Get the location where the downloaded imagery will be saved
        file_path = os.path.abspath(os.path.join(os.getcwd(), "data"))
        date_str = file_utilities.generate_datestring()
        settings = self.get_settings()

        # selected_layer contains the selected ROIs
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        # Create a list of download settings for each ROI
        roi_settings = common.create_roi_settings(
            settings, selected_layer.data, file_path, date_str
        )

        # Save the ROI settings
        self.rois.set_roi_settings(roi_settings)

        # create a list of settings for each ROI
        inputs_list = list(roi_settings.values())
        logger.info(f"inputs_list {inputs_list}")

        # Save settings used to download rois and the objects on map to config files
        self.save_config()

        # 2. For each ROI use download settings to download imagery and save to jpg
        print("Download in progress")
        # for each ROI use the ROI settings to download imagery and save to jpg
        for inputs_for_roi in tqdm(inputs_list, desc="Downloading ROIs"):
            SDS_download.retrieve_images(
                inputs_for_roi,
                cloud_threshold=settings.get("cloud_thresh"),
                cloud_mask_issue=settings.get("cloud_mask_issue"),
                save_jpg=True,
                apply_cloud_mask=settings.get("apply_cloud_mask", True),
            )
        if settings.get("image_size_filter", True):
            common.filter_images_by_roi(roi_settings)

        logger.info("Done downloading")


    def load_json_config(self, filepath: str,) -> dict:
        """
        Loads a .json configuration file specified by the user.
        Updates the coastseg_map.settings with the settings from the config file.

        Args:
            self (object): CoastsegMap instance
            filepath (str): The filepath to the json config file

        Returns:
            dict: The json data loaded from the file

        Raises:
            FileNotFoundError: If the config file is not found
            MissingDirectoriesError: If one or more directories specified in the config file are missing

        """
        logger.info(f"Loading json config from filepath: {filepath}")
        exception_handler.check_if_None(self.rois)

        json_data = file_utilities.read_json_file(filepath, raise_error=True)
        json_data = json_data or {}
        # Replace coastseg_map.settings with settings from config file
        settings = common.load_settings(
            new_settings=json_data,
        )
        self.set_settings(**settings)
        return json_data
        

    
    def load_config_files(self,  data_path: str,config_geojson_path:str, config_json_path:str) -> None:
        """Loads the configuration files from the specified directory
            Loads config_gdf.geojson first, then config.json.
        - config.json relies on config_gdf.geojson to load the rois on the map
        Args:
            data_path (str): Path to the directory where downloaded data will be saved.
            config_geojson_path (str): Path to the config_gdf.geojson file.
            config_json_path (str): Path to the config.json file.
        Raises:
            Exception: Raised if config files are missing.
        Returns:
            bool: True if both config files exist, False otherwise.
        """
        # if config_gdf.geojson does not exist, then this might be the wrong directory
        if not file_utilities.file_exists(config_geojson_path, "config_gdf.geojson"):
            return False

        # config.json contains all the settings for the map, shorelines and transects it must exist
        if not file_utilities.file_exists(config_json_path, "config.json"):
            raise Exception(f"config.json file missing at {config_json_path}")

        # load the config files
        # load general settings from config.json file
        self.load_gdf_config(config_geojson_path)
        json_data = self.load_json_config(config_json_path)
        # creates a dictionary mapping ROI IDs to their extracted settings from json_data
        roi_settings = common.process_roi_settings(json_data, data_path)
        # Make sure each ROI has the specific settings for its save location, its ID, coordinates etc.
        if hasattr(self, "rois"):
            self.rois.update_roi_settings(roi_settings)
            # self.rois.roi_settings.update(roi_settings)
        # if hasattr(self, "rois"):
        #     self.rois.roi_settings = roi_settings 
     
            
        logger.info(f"roi_settings: {roi_settings} loaded from {config_json_path}")
        
        # update the config.json files with the filepath of the data directory on this computer
        common.update_downloaded_configs(roi_settings)
        # return true if both config files exist
        return True

                
    def save_config(self, filepath: str = None,selected_only:bool=True,data_path:str=None) -> None:
        """saves the configuration settings of the map into two files
            config.json and config_gdf.geojson
            Saves the inputs such as dates, landsat_collection, satellite list, and ROIs
            Saves the settings such as preprocess settings
        Args:
            file_path (str, optional): path to directory to save config files. Defaults to None.
        Raises:
            Exception: raised if self.settings is missing
            ValueError: raised if any of "dates", "sat_list", "landsat_collection" is missing from self.settings
            Exception: raised if self.rois is missing
            Exception: raised if selected_layer is missing
        """
        settings = self.get_settings()
        # if no rois exist on the map do not allow configs to be saved
        exception_handler.config_check_if_none(self.rois, "ROIs")

        # selected_layer must contain selected ROI
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_roi_layer(selected_layer)
        logger.info(f"self.rois.roi_settings: {self.rois.roi_settings}")

        # if the rois do not have any settings then save the currently loaded settings to the ROIs
        if not self.rois.get_roi_settings():
            filepath = filepath or os.path.abspath(os.getcwd())
            roi_settings = common.create_roi_settings(
                settings, selected_layer.data, filepath
            )
            self.rois.set_roi_settings(roi_settings)

        roi_ids = self.get_roi_ids()
        if selected_only:
            # create dictionary of settings for each ROI to be saved to config.json
            roi_ids = self.get_roi_ids(is_selected=True)
        
        selected_roi_settings = {
            roi_id: self.rois.roi_settings[roi_id] for roi_id in roi_ids
        }
        # combine the settings for each ROI with the rest of the currently loaded settings
        config_json = common.create_json_config(selected_roi_settings, settings)

        shorelines_gdf = (
            getattr(self.shoreline, "gdf", None) if self.shoreline else None
        )
        transects_gdf = getattr(self.transects, "gdf", None) if self.transects else None
        bbox_gdf = getattr(self.bbox, "gdf", None) if self.bbox else None
        # get the geodataframe containing all the selected rois
        selected_rois = self.rois.gdf[self.rois.gdf["id"].isin(roi_ids)]
        logger.info(f"selected_rois: {selected_rois}")

        # save all selected rois, shorelines, transects and bbox to config geodataframe
        if selected_rois is not None:
            if not selected_rois.empty:
                epsg_code = selected_rois.crs
        config_gdf = common.create_config_gdf(
            selected_rois,
            shorelines_gdf=shorelines_gdf,
            transects_gdf=transects_gdf,
            bbox_gdf=bbox_gdf,
            epsg_code=epsg_code,
        )
        logger.info(f"config_gdf: {config_gdf} ")

        def save_config_files(config_json, config_gdf, path):
            """Helper function to save config files."""
            file_utilities.config_to_file(config_json, path)
            file_utilities.config_to_file(config_gdf, path)

        if filepath:
            # save the config.json and config_gdf.geojson immediately to the filepath directory
            save_config_files(config_json, config_gdf, filepath)
            print(f"Saved config files to {filepath}")
        else:
            is_downloaded = common.were_rois_downloaded(self.rois.get_roi_settings(), roi_ids)
            # if  data has been downloaded before then inputs have keys 'filepath' and 'sitename'
            if is_downloaded:
                # write config_json file to each directory where a roi was saved
                roi_ids = config_json["roi_ids"]
                for roi_id in roi_ids:
                    sitename = str(config_json[roi_id]["sitename"])
                    filepath = os.path.abspath(
                        os.path.join(config_json[roi_id]["filepath"], sitename)
                    )
                    save_config_files(config_json, config_gdf, filepath)
                print("Saved config files for each ROI")
            else:
                # if data is not downloaded save to coastseg directory
                filepath = os.path.abspath(os.getcwd())
                save_config_files(config_json, config_gdf, filepath)
                print(f"Saved config files for each ROI to {filepath}")

    def set_settings(self, **kwargs):
        """
        Saves the settings for downloading data by updating the `self.settings` dictionary with the provided key-value pairs.
        If any of the keys are missing, they will be set to their default value as specified in `default_settings`.

        Example: set_settings(sat_list=sat_list, dates=dates,**more_settings)

        Args:
        **kwargs: Keyword arguments representing the key-value pairs to be added to or updated in `self.settings`.

        Returns:
        None
        """
        logger.info(f"New Settings: {kwargs}")
        # logger.info(f"OLD Settings: {self.settings}")
        # Check if any of the keys are missing
        # if any keys are missing set the default value
        self.default_settings = {
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

        # Function to parse dates with flexibility for different formats
        def parse_date(date_str):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            raise ValueError(f"Date format for {date_str} not recognized.")

        # Update the settings with the new key-value pairs
        self.settings.update(kwargs)

        # Special handling for 'dates'
        if "dates" in kwargs:
            self.settings["dates"] = [parse_date(d) for d in kwargs["dates"]]

        for key, value in self.default_settings.items():
            self.settings.setdefault(key, value)

        logger.info(f"Set Settings: {self.settings}")

    def get_settings(self):
        """
        Retrieves the current settings.

        Returns:
            dict: A dictionary containing the current settings.

        Raises:
            Exception: If no settings are found. Click save settings or load a config file.

        """
        SETTINGS_NOT_FOUND = (
            "No settings found. Click save settings or load a config file."
        )
        if self.settings is None or self.settings == {}:
            raise Exception(SETTINGS_NOT_FOUND)
        return self.settings

    def update_transects_html(self, feature: dict, **kwargs):
        """
        Modifies the HTML when a transect is hovered over.

        Args:
            feature (dict): The transect feature.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        properties = feature["properties"]
        transect_id = properties.get("id", "unknown")
        slope = properties.get("slope", "unknown")
        distance = properties.get("distance", "unknown")
        feature_x = properties.get("feature_x", "unknown")
        feature_y = properties.get("feature_y", "unknown")
        nearest_x = properties.get("nearest_x", "unknown")
        nearest_y = properties.get("nearest_y", "unknown")
        variables = [distance, feature_x, feature_y, nearest_x, nearest_y]

        def is_unknown_or_None_or_nan(value):
            if isinstance(value, str):
                return True
            if not value:
                return True
            elif math.isnan(value):
                return True

        # Conditional rounding or keep as 'unknown'
        distance = (
            round(float(distance), 3)
            if not is_unknown_or_None_or_nan(distance)
            else "unknown"
        )
        feature_x = (
            round(float(feature_x), 6)
            if not is_unknown_or_None_or_nan(feature_x)
            else "unknown"
        )
        feature_y = (
            round(float(feature_y), 6)
            if not is_unknown_or_None_or_nan(feature_y)
            else "unknown"
        )
        nearest_x = (
            round(float(nearest_x), 6)
            if not is_unknown_or_None_or_nan(nearest_x)
            else "unknown"
        )
        nearest_y = (
            round(float(nearest_y), 6)
            if not is_unknown_or_None_or_nan(nearest_y)
            else "unknown"
        )

        self.feature_html.value = (
            "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
            "<b>Transect</b>"
            f"<p>Id: {transect_id}</p>"
            f"<p>Slope: {slope}</p>"
            f"<p>Distance btw slope and transect: {distance}</p>"
            f"<p>Transect (x,y):({feature_x},{feature_y})</p>"
            f"<p>Nearest Slope (x,y):({nearest_x},{nearest_y})</p>"
        )

    def update_extracted_shoreline_html(self, feature, **kwargs):
        """
        Modifies the HTML content when an extracted shoreline is hovered over.

        Args:
            feature (dict): The extracted shoreline feature.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        # Modifies html when extracted shoreline is hovered over
        properties = feature["properties"]
        date = properties.get("date", "unknown")
        cloud_cover = properties.get("cloud_cover", "unknown")
        satname = properties.get("satname", "unknown")
        geoaccuracy = properties.get("geoaccuracy", "unknown")

        self.feature_html.value = (
            "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
            "<b>Extracted Shoreline</b>"
            f"<p>Date: {date}</p>"
            f"<p>Geoaccuracy: {geoaccuracy}</p>"
            f"<p>Cloud Cover: {cloud_cover}</p>"
            f"<p>Satellite Name: {satname}</p>"
        )

    def update_roi_html(self, feature, **kwargs):
        # Modifies html when roi is hovered over
        values = defaultdict(lambda: "unknown", feature["properties"])
        self.roi_html.value = """ 
        <div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>
        <b>ROI</b>
        <p>Id: {}</p>
        """.format(
            values["id"],
        )

    def update_shoreline_html(self, feature, **kwargs):
        """
        Modifies the HTML content when a shoreline is hovered over.

        Args:
            feature (dict): The shoreline feature.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        # Modifies html when shoreline is hovered over
        properties = feature["properties"]
        shoreline_id = properties.get("id", "unknown")
        mean_sig_waveheight = properties.get("MEAN_SIG_WAVEHEIGHT", "unknown")
        tidal_range = properties.get("TIDAL_RANGE", "unknown")
        erodibility = properties.get("ERODIBILITY", "unknown")
        river_label = properties.get("river_label", "unknown")
        sinuosity_label = properties.get("sinuosity_label", "unknown")
        slope_label = properties.get("slope_label", "unknown")
        turbid_label = properties.get("turbid_label", "unknown")
        csu_id = properties.get("CSU_ID", "unknown")

        self.feature_html.value = (
            "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
            "<b>Shoreline</b>"
            f"<p>ID: {shoreline_id}</p>"
            f"<p>Mean Sig Waveheight: {mean_sig_waveheight}</p>"
            f"<p>Tidal Range: {tidal_range}</p>"
            f"<p>Erodibility: {erodibility}</p>"
            f"<p>River: {river_label}</p>"
            f"<p>Sinuosity: {sinuosity_label}</p>"
            f"<p>Slope: {slope_label}</p>"
            f"<p>Turbid: {turbid_label}</p>"
            f"<p>CSU_ID: {csu_id}</p>"
        )

    def get_all_roi_ids(self) -> List[str]:
        """
        Return a list of all ROI IDs.

        Args:
            None.

        Returns:
            A list of all ROI IDs.

        Raises:
            None.
        """
        if self.rois is None:
            return []
        if self.rois.gdf.empty:
            return []
        if not hasattr(self.rois, "gdf"):
            return []
        if "id" not in self.rois.gdf.columns:
            return []
        return self.rois.gdf["id"].tolist()

    def get_any_available_roi_id(self) -> List[str]:
        roi_ids = self.get_roi_ids(is_selected=True)
        if roi_ids == []:
            roi_ids = self.get_all_roi_ids()
            if roi_ids == []:
                return roi_ids
            roi_ids = roi_ids[0]
        return roi_ids

    def load_extracted_shoreline_files(self) -> None:
        """
        Loads extracted shoreline files for each ROI and adds them to the map.
        Raises:
            Exception: If no extracted shorelines could be loaded.
        """
        exception_handler.config_check_if_none(self.rois, "ROIs")
        # load extracted shorelines for either a selected ROI or the first ROI if no ROI is selected
        roi_ids = self.get_any_available_roi_id()
        # set of roi ids that didn't have missing shorelines
        rois_no_extracted_shorelines = set()
        # for each ROI that has extracted shorelines load onto map
        for roi_id in roi_ids:
            filepath = self.rois.roi_settings[roi_id]["filepath"]
            sitename = self.rois.roi_settings[roi_id]["sitename"]
            roi_path = os.path.join(filepath, sitename)
            glob_str = os.path.abspath(roi_path + os.sep + "*shoreline*")
            extracted_sl_gdf = None
            shoreline_settings = None
            extracted_shoreline_dict = None
            for file in glob.glob(glob_str):
                if file.endswith(".geojson"):
                    # load geodataframe
                    extracted_sl_gdf = geodata_processing.read_gpd_file(file)
                if file.endswith(".json"):
                    if "settings" in os.path.basename(file):
                        shoreline_settings = file_utilities.load_data_from_json(file)
                    if "dict" in os.path.basename(file):
                        extracted_shoreline_dict = file_utilities.load_data_from_json(
                            file
                        )
            # If any of the extracted shoreline files are missing, skip to next ROI
            if extracted_sl_gdf is None:
                logger.info(
                    f"ROI {roi_id} didn't have extracted shoreline files to load"
                )
                rois_no_extracted_shorelines.add(roi_id)
                # set this roi's entry in extracted shorelines dict to None because there was no shoreline extracted
                self.rois.add_extracted_shoreline(None, roi_id)
                continue
            else:
                extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
                extracted_shorelines = extracted_shorelines.load_extracted_shorelines(
                    extracted_shoreline_dict, shoreline_settings, extracted_sl_gdf
                )
                self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)
                logger.info(
                    f"ROI {roi_id} successfully loaded extracted shorelines: {self.rois.get_extracted_shoreline(roi_id)}"
                )
        if len(rois_no_extracted_shorelines) > 0:
            logger.warning(
                f"The following ROIs didn't have extracted shoreline files to load: {rois_no_extracted_shorelines}\n"
            )
        rois_with_shorelines = set(roi_ids) - rois_no_extracted_shorelines
        if len(rois_with_shorelines) > 0:
            print(f"Loaded Extracted Shorelines for ROIs {rois_with_shorelines}")
        elif len(rois_with_shorelines) == 0:
            raise Exception("No extracted shorelines could be loaded")

    def extract_shoreline_for_roi(
        self,
        roi_id: str,
        rois_gdf: gpd.GeoDataFrame,
        shoreline_gdf: gpd.GeoDataFrame,
        settings: dict,
    ) -> Optional[extracted_shoreline.Extracted_Shoreline]:
        """
        Extracts the shoreline for a given ROI and returns the extracted shoreline object.

        Parameters:
        - roi_id (str): the ID of the ROI to extract the shoreline from
        - rois_gdf (geopandas.GeoDataFrame): the GeoDataFrame containing all the ROIs
        - shoreline_gdf (geopandas.GeoDataFrame): the GeoDataFrame containing the shoreline
        - settings (dict): the settings to use for the shoreline extraction


        Returns:
        - extracted_shoreline.Extracted_Shoreline or None: the extracted shoreline object for the ROI, or None if an error occurs

        Raises:
        - No exceptions are raised by this function.
        """
        try:
            logger.info(f"Extracting shorelines from ROI with the id: {roi_id}")
            roi_settings = self.rois.get_roi_settings(roi_id)
            single_roi = common.extract_roi_by_id(rois_gdf, roi_id)
            # Clip shoreline to specific roi
            shoreline_in_roi = gpd.clip(shoreline_gdf, single_roi)
            # extract shorelines from ROI
            extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
            extracted_shorelines = extracted_shorelines.create_extracted_shorelines(
                roi_id,
                shoreline_in_roi,
                roi_settings,
                settings,
            )
            logger.info(f"extracted_shoreline_dict[{roi_id}]: {extracted_shorelines}")
            return extracted_shorelines
        except exceptions.Id_Not_Found as id_error:
            logger.warning(
                f"exceptions.Id_Not_Found {id_error} {traceback.format_exc()}"
            )
            print(f"ROI with id {roi_id} was not found. \n Skipping to next ROI")
        except exceptions.No_Extracted_Shoreline as no_shoreline:
            logger.warning(f"{roi_id}: {no_shoreline} {traceback.format_exc()}")
            print(f"{roi_id}: {no_shoreline}")
        except Exception as e:
            logger.warning(
                f"An error occurred while extracting shoreline for ROI {roi_id}: {e} \n {traceback.format_exc()}"
            )
            print(
                f"An error occurred while extracting shoreline for ROI {roi_id}. \n Skipping to next ROI \n {e} \n {traceback.format_exc()}"
            )
        return None

    def update_settings_with_accurate_epsg(self):
        """Updates settings with the most accurate epsg code based on lat and lon if output epsg
        was 4326 or 4327.
        """
        settings = self.get_settings()
        new_espg = common.get_most_accurate_epsg(
            settings.get("output_epsg", 4326), self.bbox.gdf
        )
        self.set_settings(output_epsg=new_espg)

    def validate_transect_inputs(self, settings):
        # ROIs,settings, roi-settings cannot be None or empty
        exception_handler.check_if_empty_string(self.get_session_name(), "session name")
        # ROIs, transects, and extracted shorelines must exist
        exception_handler.check_if_None(self.rois, "ROIs")
        exception_handler.check_if_None(self.transects, "transects")
        exception_handler.check_empty_dict(
            self.rois.get_all_extracted_shorelines(), "extracted_shorelines"
        )
        # settings must contain key 'along_dist'
        exception_handler.check_if_subset(
            set(["along_dist"]), set(list(settings.keys())), "settings"
        )
        # if no rois are selected throw an error
        exception_handler.check_selected_set(self.selected_set)

        # ids of ROIs that have had their shorelines extracted
        extracted_shoreline_ids = set(
            list(self.rois.get_all_extracted_shorelines().keys())
        )
        # Get ROI ids that are selected on map and have had their shorelines extracted
        roi_ids = list(extracted_shoreline_ids & self.selected_set)
        # if none of the selected ROIs on the map have had their shorelines extracted throw an error
        exception_handler.check_if_list_empty(roi_ids)



    def validate_extract_shoreline_inputs(self):
        # ROIs,settings, roi-settings cannot be None or empty
        settings = self.get_settings()
        exception_handler.check_if_empty_string(self.get_session_name(), "session name")
        # ROIs, transects,shorelines and a bounding box must exist
        exception_handler.validate_feature(self.rois, "roi")
        exception_handler.validate_feature(self.shoreline, "shoreline")
        exception_handler.validate_feature(self.transects, "transects")
        exception_handler.validate_feature(self.bbox, "bounding box")
        # ROI settings must not be empty
        if hasattr(self.rois, "roi_settings"):
            exception_handler.check_empty_dict(self.rois.roi_settings, "roi_settings")
        else:
            raise Exception("None of the ROIs have been downloaded on this machine or the location where they were downloaded has been moved. Please download the ROIs again.")

        # settings must contain keys "dates", "sat_list", "landsat_collection"
        superset = set(list(settings.keys()))
        exception_handler.check_if_subset(
            set(["dates", "sat_list", "landsat_collection"]), superset, "settings"
        )

        # if no rois are selected throw an error
        exception_handler.check_selected_set(self.selected_set)

        # roi_settings must contain roi ids in selected set
        superset = set(list(self.rois.roi_settings.keys()))
        error_message = "To extract shorelines you must first select ROIs and have the data downloaded."
        exception_handler.check_if_subset(
            self.selected_set, superset, "roi_settings", error_message
        )
        # get only the rois with missing directories that are selected on the map
        roi_ids = self.get_roi_ids(is_selected=True)
        # check if any of the ROIs are missing their downloaded data directory
        missing_directories = common.get_missing_roi_dirs(self.rois.get_roi_settings(),roi_ids)
        # raise an warning if any of the selected ROIs were not downloaded 
        exception_handler.check_if_dirs_missing(missing_directories)
        

    def validate_download_imagery_inputs(self):
        """
        Validates the inputs required for downloading imagery.

        This method checks if the necessary settings are present and if the selected layer contains the selected ROI.

        Raises:
            SubsetError: If the required settings keys are not present in the settings.
            EmptyLayerError: If the selected layer is empty.
            EmptyROILayerError: If the selected layer does not contain any ROI.
        """
        # settings cannot be None
        settings = self.get_settings()
        # Ensure the required keys are present in the settings
        required_settings_keys = set(["dates", "sat_list", "landsat_collection"])
        superset = set(list(settings.keys()))
        exception_handler.check_if_subset(required_settings_keys, superset, "settings")

        # selected_layer must contain selected ROI
        selected_layer = self.map.find_layer(ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_layer(selected_layer, ROI.SELECTED_LAYER_NAME)
        exception_handler.check_empty_roi_layer(selected_layer)

    def get_roi_ids(
        self, is_selected: bool = False, has_shorelines: bool = False
    ) -> list:
        """
        Get the IDs of the regions of interest (ROIs) that meet the specified criteria.

        Args:
            is_selected (bool, optional): Whether to consider only the selected ROIs on the map. Defaults to True.
            has_shorelines (bool, optional): Whether to consider only the ROIs that have extracted shorelines. Defaults to False.

        Returns:
            list: The IDs of the ROIs that meet the specified criteria.
        """
        roi_ids = self.get_all_roi_ids()
        if has_shorelines:
            roi_ids = set(self.rois.get_ids_with_extracted_shorelines())
        if is_selected:
            roi_ids = list(set(roi_ids) & self.selected_set)
        return roi_ids

    def extract_all_shorelines(self) -> None:
        """
        Extracts shorelines for all selected regions of interest (ROIs).

        This method performs the following steps:
        1. Validates the inputs for shoreline extraction.
        2. Retrieves the IDs of the selected ROIs.
        3. Updates the settings with the most accurate EPSG.
        4. Saves the updated configurations.
        5. Extracts shorelines for each selected ROI.
        6. Saves the ROI IDs that had extracted shorelines.
        7. Saves a session for each ROI.
        8. Computes transects for selected ROIs with extracted shorelines.
        9. Loads extracted shorelines to the map.
        10. Updates the available ROI IDs with extracted shorelines.

        Note: This method assumes that the necessary data structures and attributes are already initialized.

        Returns:
            None
        """
        # 1. validate the inputs for shoreline extraction exist: ROIs, transects,shorelines and a downloaded data for each ROI
        self.validate_extract_shoreline_inputs()
        # if self.session_exists(self.get_session_name()):
        #     raise Exception("Session already exists. Please save the session with a different name.")
        # if the session name of where the extracted shorelines already exists
        # then don't allow the user to extract shorelines to this session 
        roi_ids = self.get_roi_ids(is_selected=True)
        logger.info(f"roi_ids to extract shorelines from: {roi_ids}")
        #2. update the settings with the most accurate epsg
        self.update_settings_with_accurate_epsg()
    
        # @todo make this change official after finding a way to test it properly    
        # save the updated configs
        # session_name = self.get_session_name()
        # for roi_id in roi_ids:
        #     # name of the directory where the extracted shorelines will be saved under the session name
        #     ROI_directory = self.rois.roi_settings[roi_id]["sitename"]
        #     session_path = file_utilities.create_session_path(
        #                     session_name, ROI_directory
        #                 )
        #     self.save_config(session_path)
        
        #3. get selected ROIs on map and extract shoreline for each of them
        for roi_id in tqdm(roi_ids, desc="Extracting Shorelines"):
            print(f"Extracting shorelines from ROI with the id:{roi_id}")
            extracted_shorelines = self.extract_shoreline_for_roi(
                roi_id, self.rois.gdf, self.shoreline.gdf, self.get_settings()
            )
            self.rois.add_extracted_shoreline(extracted_shorelines, roi_id)

        #4. save the ROI IDs that had extracted shoreline to observable variable roi_ids_with_extracted_shorelines
        ids_with_extracted_shorelines = self.get_roi_ids(
            is_selected=False, has_shorelines=True
        )
        # update the available ROI IDs and this will update the extracted shorelines on the map
        if ids_with_extracted_shorelines is None:
            self.id_container.ids = []
        elif not isinstance(ids_with_extracted_shorelines, list):
            self.id_container.ids = list(ids_with_extracted_shorelines)
        else:
            self.id_container.ids = ids_with_extracted_shorelines

        #4. save a session for each ROI under one session name
        self.save_session(roi_ids, save_transects=False)

        #6. Get ROI ids that are selected on map and have had their shorelines extracted, and compute transects for them
        roi_ids = self.get_roi_ids(is_selected=True, has_shorelines=True)
        if hasattr(self.transects, "gdf"):
            self.compute_transects(self.transects.gdf, self.get_settings(), roi_ids)
        # load extracted shorelines to map
        # update the available ROI IDs and this will update the extracted shorelines on the map
        ids_with_extracted_shorelines = self.update_roi_ids_with_shorelines()
        logger.info(
            f"Available roi_ids from extracted shorelines: {ids_with_extracted_shorelines}"
        )

    def get_cross_distance(
        self,
        roi_id: str,
        transects_in_roi_gdf: gpd.GeoDataFrame,
        settings: dict,
        output_epsg: int,
    ) -> Tuple[float, Optional[str]]:
        """
        Compute the cross shore distance of transects and extracted shorelines for a given ROI.

        Parameters:
        -----------
        roi_id : str
            The ID of the ROI to compute the cross shore distance for.
        transects_in_roi_gdf : gpd.GeoDataFrame
            All the transects in the ROI. Must contain the columns ["id", "geometry"]
        settings : dict
            A dictionary of settings to be used in the computation.
        output_epsg : int
            The EPSG code of the output projection.

        Returns:
        --------
        Tuple[float, Optional[str]]
            The computed cross shore distance, or 0 if there was an issue in the computation.
            The reason for failure, or '' if the computation was successful.
        """
        failure_reason = ""
        cross_distance = 0

        # Get extracted shorelines object for the currently selected ROI
        roi_extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)

        transects_in_roi_gdf = transects_in_roi_gdf.loc[:, ["id", "geometry"]]

        if roi_extracted_shoreline is None:
            failure_reason = "No extracted shorelines were found"

        elif transects_in_roi_gdf.empty:
            failure_reason = "No transects intersect"

        else:
            # Convert transects_in_roi_gdf to output_crs from settings
            transects_in_roi_gdf = transects_in_roi_gdf.to_crs(output_epsg)
            # Compute cross shore distance of transects and extracted shorelines
            extracted_shorelines_dict = roi_extracted_shoreline.dictionary
            cross_distance = extracted_shoreline.compute_transects_from_roi(
                extracted_shorelines_dict,
                transects_in_roi_gdf,
                settings,
            )
            if cross_distance == 0:
                failure_reason = "Cross distance computation failed"

        return cross_distance, failure_reason

    def compute_transects(
        self, transects_gdf: gpd.GeoDataFrame, settings: dict, roi_ids: list[str]
    ) -> dict:
        """Returns a dict of cross distances for each roi's transects
        Args:
            selected_rois (dict): rois selected by the user. Must contain the following fields:
                {'features': [
                    'id': (str) roi_id
                    'geometry':{
                        'type':Polygon
                        'coordinates': list of coordinates that make up polygon
                    }
                ],...}
            extracted_shorelines (dict): dictionary with roi_id keys that identify roi associates with shorelines
            {
                roi_id:{
                    dates: [datetime.datetime,datetime.datetime], ...
                    shorelines: [array(),array()]    }
            }

            settings (dict): settings used for CoastSat. Must have the following fields:
               'output_epsg': int
                    output spatial reference system as EPSG code
                'along_dist': int
                    alongshore distance considered caluclate the intersection

        Returns:
            dict: cross_distances_rois with format:
            { roi_id :  dict
                time-series of cross-shore distance along each of the transects. Not tidally corrected. }
        """
        self.validate_transect_inputs(settings)
        # user selected output projection
        output_epsg = "epsg:" + str(settings["output_epsg"])
        # for each ROI save cross distances for each transect that intersects each extracted shoreline
        for roi_id in tqdm(roi_ids, desc="Computing Cross Distance Transects"):
            # get transects that intersect with ROI
            single_roi = common.extract_roi_by_id(self.rois.gdf, roi_id)
            # save cross distances by ROI id
            transects_in_roi_gdf = transects_gdf[
                transects_gdf.intersects(single_roi.unary_union)
            ]
            cross_distance, failure_reason = self.get_cross_distance(
                str(roi_id), transects_in_roi_gdf, settings, output_epsg
            )
            if cross_distance == 0:
                logger.warning(f"{failure_reason} for ROI {roi_id}")
                print(f"{failure_reason} for ROI {roi_id}")
            self.rois.add_cross_shore_distances(cross_distance, roi_id)

        self.save_session(roi_ids)

    def session_exists(self, session_name: str) -> bool:
            """
            Check if a session with the given name exists.

            Args:
                session_name (str): The name of the session to check.

            Returns:
                bool: True if the session exists, False otherwise.
            """
            session_name = self.get_session_name()
            session_path = os.path.join(os.getcwd(), "sessions", session_name)
            if os.path.exists(session_path):
                # check if session directory contains a directory with the roi_id
                dirs=os.listdir(session_path)
                if dirs:
                    return True
                else:
                    return False
            return False

      
    def save_session(self, roi_ids: list[str], save_transects: bool = True):
        # Save extracted shoreline info to session directory
        session_name = self.get_session_name()
        for roi_id in roi_ids:
            ROI_directory = self.rois.roi_settings[roi_id]["sitename"]
            # create session directory
            session_path = file_utilities.create_session_path(
                session_name, ROI_directory
            )
            # save source data
            self.save_config(session_path)
            # save extracted shorelines
            extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)
            logger.info(f"Extracted shorelines for ROI {roi_id}: {extracted_shoreline}")
            if extracted_shoreline is None:
                logger.info(f"No extracted shorelines for ROI: {roi_id}")
                continue
            # move extracted shoreline figures to session directory
            shoreline_settings = extracted_shoreline.shoreline_settings
            common.save_extracted_shoreline_figures(shoreline_settings, session_path)
            # move extracted shoreline reports to session directory
            common.move_report_files(
                shoreline_settings, session_path, "extract_shorelines*.txt"
            )
            # save the geojson and json files for extracted shorelines
            common.save_extracted_shorelines(extracted_shoreline, session_path)

            # save transects to session folder
            if save_transects:
                # get extracted_shorelines from extracted shoreline object in rois
                extracted_shorelines_dict = extracted_shoreline.dictionary
                # if no shorelines were extracted then skip
                if extracted_shorelines_dict == {}:
                    logger.info(f"No extracted shorelines for roi: {roi_id}")
                    continue
                cross_shore_distance = self.rois.get_cross_shore_distances(roi_id)
                # if no cross distance was 0 then skip
                if cross_shore_distance == 0:
                    print(
                        f"ROI: {roi_id} had no time-series of shoreline change along transects"
                    )
                    logger.info(f"ROI: {roi_id} cross distance is 0")
                    continue

                common.save_transects(
                    roi_id,
                    session_path,
                    cross_shore_distance,
                    extracted_shorelines_dict,
                    self.get_settings(),
                )

    def remove_all(self):
        """Remove the bbox, shoreline, all rois from the map"""
        self.remove_bbox()
        self.remove_shoreline()
        self.remove_transects()
        self.remove_all_rois()
        self.remove_layer_by_name("geodataframe")
        self.remove_extracted_shorelines()
        # Clear the list of ROI IDs that have extracted shorelines available

    def remove_extracted_shorelines(self):
        """Removes all extracted shorelines from the map and removes extracted shorelines from ROIs"""
        # empty extracted shorelines dictionary
        if self.rois is not None:
            self.rois.remove_extracted_shorelines(remove_all=True)
        # remove extracted shoreline vectors from the map
        self.remove_extracted_shoreline_layers()
        self.id_container.ids = []
        self.extract_shorelines_container.clear()

    def remove_extracted_shoreline_layers(self):
        self.remove_layer_by_name("delete")
        self.remove_layer_by_name("extracted shoreline")

    def remove_bbox(self):
        """Remove all the bounding boxes from the map"""
        if self.bbox is not None:
            del self.bbox
            self.bbox = None
        self.draw_control.clear()
        existing_layer = self.map.find_layer(Bounding_Box.LAYER_NAME)
        if existing_layer is not None:
            self.map.remove_layer(existing_layer)
        self.bbox = None

    def remove_layer_by_name(self, layer_name: str):
        existing_layer = self.map.find_layer(layer_name)
        if existing_layer is not None:
            self.map.remove(existing_layer)

    def remove_shoreline(self):
        del self.shoreline
        self.remove_layer_by_name(Shoreline.LAYER_NAME)
        self.shoreline = None

    def remove_transects(self):
        del self.transects
        self.transects = None
        self.remove_layer_by_name(Transects.LAYER_NAME)

    def replace_layer_by_name(
        self, layer_name: str, new_layer: GeoJSON, on_hover=None, on_click=None
    ) -> None:
        """Replaces layer with layer_name with new_layer on map. Adds on_hover and on_click callable functions
        as handlers for hover and click events on new_layer
        Args:
            layer_name (str): name of layer to replace
            new_layer (GeoJSON): ipyleaflet GeoJSON layer to add to map
            on_hover (callable, optional): Callback function that will be called on hover event on a feature, this function
            should take the event and the feature as inputs. Defaults to None.
            on_click (callable, optional): Callback function that will be called on click event on a feature, this function
            should take the event and the feature as inputs. Defaults to None.
        """
        if new_layer is None:
            return
        self.remove_layer_by_name(layer_name)
        # when feature is hovered over on_hover function is called
        if on_hover is not None:
            new_layer.on_hover(on_hover)
        if on_click is not None:
            # when feature is clicked on on_click function is called
            new_layer.on_click(on_click)
        self.map.add_layer(new_layer)

    def remove_all_rois(self) -> None:
        """Removes all the unselected rois from the map"""
        # Remove the selected and unselected rois
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        self.remove_layer_by_name(ROI.LAYER_NAME)
        # clear all the ids from the selected set
        self.selected_set = set()
        del self.rois
        self.rois = None

    def remove_selected_shorelines(self) -> None:
        """Removes all the unselected shorelines from the map"""
        logger.info("Removing selected shorelines from map")
        # Remove the selected and unselected rois
        self.remove_layer_by_name(SELECTED_LAYER_NAME)
        self.remove_layer_by_name(Shoreline.LAYER_NAME)
        # delete selected ROIs from dataframe
        if self.shoreline:
            self.shoreline.remove_by_id(self.selected_shorelines_set)
        # clear all the ids from the selected set
        self.selected_shorelines_set = set()
        # reload rest of shorelines on map
        if hasattr(self.shoreline, "gdf"):
            self.load_feature_on_map(
                "shoreline", gdf=self.shoreline.gdf, zoom_to_bounds=True
            )

    def remove_selected_rois(self) -> None:
        """Removes all the unselected rois from the map"""
        # Remove the selected and unselected rois
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        self.remove_layer_by_name(ROI.LAYER_NAME)
        # delete selected ROIs from dataframe
        if self.rois:
            self.rois.remove_by_id(self.selected_set)
        # clear all the ids from the selected set
        self.selected_set = set()
        # reload rest of ROIs on map
        if hasattr(self.rois, "gdf"):
            self.load_feature_on_map("roi", gdf=self.rois.gdf, zoom_to_bounds=True)

    def create_DrawControl(self, draw_control: DrawControl) -> DrawControl:
        """Modifies given draw control so that only rectangles can be drawn

        Args:
            draw_control (ipyleaflet.leaflet.DrawControl): draw control to modify

        Returns:
            ipyleaflet.leaflet.DrawControl: modified draw control with only ability to draw rectangles
        """
        draw_control.polyline = {}
        draw_control.circlemarker = {}
        draw_control.polygon = {}
        draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "green",
                "color": "green",
                "fillOpacity": 0.1,
                "Opacity": 0.1,
            },
            "drawError": {"color": "#dd253b", "message": "Ops!"},
            "allowIntersection": False,
            "transform": True,
        }
        return draw_control

    def handle_draw(self, target: DrawControl, action: str, geo_json: dict):
        """Adds or removes the bounding box  when drawn/deleted from map
        Args:
            target (ipyleaflet.leaflet.DrawControl): draw control used
            action (str): name of the most recent action ex. 'created', 'deleted'
            geo_json (dict): geojson dictionary
        """
        if (
            self.draw_control.last_action == "created"
            and self.draw_control.last_draw["geometry"]["type"] == "Polygon"
        ):
            # validate the bbox size
            geometry = self.draw_control.last_draw["geometry"]
            bbox_area = common.get_area(geometry)
            try:
                Bounding_Box.check_bbox_size(bbox_area)
            except exceptions.BboxTooLargeError as bbox_too_big:
                self.remove_bbox()
                exception_handler.handle_bbox_error(bbox_too_big, self.warning_box)
            except exceptions.BboxTooSmallError as bbox_too_small:
                self.remove_bbox()
                exception_handler.handle_bbox_error(bbox_too_small, self.warning_box)
            else:
                # if no exceptions occur create new bbox, remove old bbox, and load new bbox
                self.load_feature_on_map("bbox")

        if self.draw_control.last_action == "deleted":
            self.remove_bbox()

    def load_extracted_shoreline_by_id(self, selected_id: str, row_number: int = 0):
        """
        Loads extracted shorelines onto a map for a single region of interest specified by its ID.

        Args:
            selected_id (str): The ID of the region of interest to plot extracted shorelines for.
            row_number (int, optional): The row number of the region of interest to plot. Defaults to 0.
        """
        # remove any existing extracted shorelines
        self.remove_extracted_shoreline_layers()
        # get the extracted shorelines for the selected roi
        if self.rois is not None:
            extracted_shorelines = self.rois.get_extracted_shoreline(selected_id)
            # logger.info(
            #     f"ROI ID { selected_id} extracted shorelines {extracted_shorelines}"
            # )
            # if extracted shorelines exist, load them onto map, if none exist nothing loads
            self.load_extracted_shorelines_on_map(extracted_shorelines, row_number)

    def update_roi_ids_with_shorelines(self) -> list[str]:
        """
        Returns a list of the ROI IDs with extracted shorelines and updates the id_container.ids and the extract_shorelines_container.roi_ids_list with the ROI IDs that have extracted shorelines.

        Updates the id_container ids and the extract_shorelines_container.roi_ids_list with the ROI IDs that have extracted shorelines.

        Returns:
            A list of ROI IDs that have extracted shorelines.
        """
        # Get the list of the ROI IDs that have extracted shorelines
        ids_with_extracted_shorelines = self.get_roi_ids(has_shorelines=True)
        logger.info(f"ids_with_extracted_shorelines: {ids_with_extracted_shorelines}")
        # if no ROIs have extracted shorelines, return otherwise load extracted shorelines for the first ROI ID with extracted shorelines
        if not ids_with_extracted_shorelines:
            self.id_container.ids = []
            self.extract_shorelines_container.roi_ids_list = []
            self.extract_shorelines_container.load_list = []
            self.extract_shorelines_container.trash_list = []
            logger.warning("No ROIs found with extracted shorelines.")
            return []
        # save the ROI IDs that had extracted shoreline to observable variables self.id_container.ids and self.extract_shorelines_container.roi_ids_list
        self.id_container.ids = list(ids_with_extracted_shorelines)
        self.extract_shorelines_container.roi_ids_list = list(
            ids_with_extracted_shorelines
        )
        return ids_with_extracted_shorelines

    def update_loadable_shorelines(
        self, selected_id: str
    ) -> extracted_shoreline.Extracted_Shoreline:
        """
        Update the loadable shorelines based on the selected ROI ID.

        Args:
            selected_id (str): The ID of the selected ROI.

        Returns:
            extracted_shorelines(extracted_shoreline.Extracted_Shoreline): The extracted shorelines for the selected ROI.
        """
        # get the extracted shoreline for the selected roi's id
        if self.rois is None:
            self.extract_shorelines_container.load_list = []
            self.extract_shorelines_container.trash_list = []
            return None

        extracted_shorelines = self.rois.get_extracted_shoreline(selected_id)
        # if extracted shorelines exist, load them onto map, if none exist nothing loads
        if hasattr(extracted_shorelines, "gdf"):
            # sort the extracted shoreline gdf by date
            if not extracted_shorelines.gdf.empty:
                extracted_shorelines.gdf = extracted_shorelines.gdf.sort_values(
                    by=["date"]
                )
                if extracted_shorelines.gdf["date"].dtype == "object":
                    # If the "date" column is already of string type, concatenate directly
                    formatted_dates = extracted_shorelines.gdf["date"]
                else:
                    # If the "date" column is not of string type, convert to string with the required format
                    formatted_dates = extracted_shorelines.gdf["date"].apply(
                        lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
                    )
                self.extract_shorelines_container.load_list = []
                self.extract_shorelines_container.load_list = (
                    extracted_shorelines.gdf["satname"] + "_" + formatted_dates
                ).tolist()
                self.extract_shorelines_container.trash_list = []
        else:
            logger.warning(f"No shorelines extracted for ROI {selected_id}")
            # if the selected ROI has no extracted shorelines, clear the load list & trash list
            self.extract_shorelines_container.load_list = []
            self.extract_shorelines_container.trash_list = []
            return None
        return extracted_shorelines

    def load_extracted_shorelines_on_map(
        self,
        extracted_shorelines: extracted_shoreline.Extracted_Shoreline,
        row_number: int = 0,
    ):
        """
        Loads a stylized extracted shoreline layer onto a map for a single region of interest.

        Args:
            extracted_shoreline (Extracted_Shoreline): An instance of the Extracted_Shoreline class containing the extracted shoreline data.
            row_number (int, optional): The row number of the region of interest to plot. Defaults to 0.
        """
        if extracted_shorelines is None:
            return
        # create the extracted shoreline layer and add it to the map
        layer_name = "extracted shoreline"
        if extracted_shorelines.gdf.empty:
            logger.info(
                f"No extracted shorelines for ROI {extracted_shorelines.roi_id}"
            )
            return
        # check if row number exists in gdf
        if row_number >= len(extracted_shorelines.gdf):
            logger.warning(
                f"Row number {row_number} does not exist in extracted shoreline gdf using row number 0 instead"
            )
            row_number = 0
        # load the selected extracted shoreline layer onto the map
        self.load_extracted_shoreline_layer(
            extracted_shorelines.gdf.iloc[[row_number]], layer_name, colormap="viridis"
        )

    def load_feature_on_map(
        self, feature_name: str, file: str = "", gdf: gpd.GeoDataFrame = None, **kwargs
    ) -> None:
        """Loads feature of type feature_name onto the map either from a file or from a geodataframe given by gdf

        if feature_name given is not one "shoreline","transects","bbox", or "rois" throw exception

        Args:
            feature_name (str): name of feature must be one of the following
            "shoreline","transects","bbox","rois"
            file (str, optional): geojson file containing feature. Defaults to "".
            gdf (gpd.GeoDataFrame, optional): geodataframe containing feature geometry. Defaults to None.
        """
        # Load GeoDataFrame if file is provided
        if file:
            gdf = geodata_processing.load_geodataframe_from_file(
                file, feature_type=feature_name
            )
        # Ensure the gdf is not empty
        if gdf is not None and gdf.empty:
            logger.info(f"No {feature_name} was loaded on map")
            return

        # create the feature
        new_feature = self.factory.make_feature(self, feature_name, gdf, **kwargs)
        if new_feature is None:
            return

        # load the features onto the map
        self.add_feature_on_map(
            new_feature,
            feature_name,
            **kwargs,
        )

    def add_feature_on_map(
        self,
        new_feature,
        feature_name: str,
        layer_name: str = "",
        zoom_to_bounds=False,
        **kwargs,
    ) -> None:
        """
        Adds a feature to the map as well as the feature's on_click and on_hover handlers.

        Args:
        - new_feature: The feature to be added to the map.
        - feature_name (str): A string representing the name of the feature.
        - layer_name (str): A string representing the name of the layer to which the feature should be added. Default value is an empty string.

        Returns:
        - None
        """
        # get on hover and on click handlers for feature
        on_hover = self.get_on_hover_handler(feature_name)
        on_click = self.get_on_click_handler(feature_name)
        # if layer name is not given use the layer name of the feature
        if not layer_name and hasattr(new_feature, "LAYER_NAME"):
            layer_name = new_feature.LAYER_NAME
        # if the feature has a geodataframe zoom the map to the bounds of the feature
        # if zoom_to_bounds and hasattr(new_feature, "gdf"):
        #     bounds = new_feature.gdf.total_bounds
        #     self.map.zoom_to_bounds(bounds)
        if hasattr(new_feature, "gdf"):
            bounds = new_feature.gdf.total_bounds
            self.map.zoom_to_bounds(bounds)
        self.load_on_map(new_feature, layer_name, on_hover, on_click)

    def get_on_click_handler(self, feature_name: str) -> callable:
        """
        Returns a callable function that handles mouse click events for a given feature.

        Args:
        - feature_name (str): A string representing the name of the feature for which the click handler needs to be returned.

        Returns:
        - callable: A callable function that handles mouse click events for a given feature.
        """
        on_click = None
        if "roi" in feature_name.lower():
            on_click = self.geojson_onclick_handler
        elif "shoreline" in feature_name.lower():
            on_click = self.shoreline_onclick_handler
        return on_click

    def get_on_hover_handler(self, feature_name: str) -> callable:
        """
        Returns a callable function that handles mouse hover events for a given feature.

        Args:
        - feature_name (str): A string representing the name of the feature for which the hover handler needs to be returned.

        Returns:
        - callable: A callable function that handles mouse hover events for a given feature.
        """
        on_hover = None
        feature_name_lower = feature_name.lower()
        if "shoreline" in feature_name_lower:
            on_hover = self.update_shoreline_html
        elif "transect" in feature_name_lower:
            on_hover = self.update_transects_html
        elif "roi" in feature_name_lower:
            on_hover = self.update_roi_html
        return on_hover

    def load_on_map(
        self, feature, layer_name: str, on_hover=None, on_click=None
    ) -> None:
        """Loads feature on map as a new layer

        Replaces current feature layer on map with given feature

        Raises:
            Exception: raised if feature layer is empty
        """
        # style and add the feature to the map
        new_layer = self.create_layer(feature, layer_name)
        # Replace old feature layer with new feature layer
        self.replace_layer_by_name(
            layer_name, new_layer, on_hover=on_hover, on_click=on_click
        )

    def create_layer(self, feature, layer_name: str):
        if feature.gdf.empty:
            logger.warning("Cannot add an empty geodataframe layer to the map.")
            print("Cannot add an empty layer to the map.")
            return None
        layer_geojson = json.loads(feature.gdf.to_json())
        # convert layer to GeoJson and style it accordingly
        styled_layer = feature.style_layer(layer_geojson, layer_name)
        return styled_layer

    def geojson_onclick_handler(
        self, event: str = None, id: str = None, properties: dict = None, **args
    ):
        """On click handler for when unselected geojson is clicked.

        Adds geojson's id to selected_set. Replaces current selected layer with a new one that includes
        recently clicked geojson.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked geojson. Defaults to None.
        """
        if properties is None:
            return
        # Add id of clicked ROI to selected_set
        self.selected_set.add(str(properties["id"]))
        # remove old selected layer
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_set, ROI.LAYER_NAME
            ),
            name=ROI.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        self.replace_layer_by_name(
            ROI.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_onclick_handler,
            on_hover=self.update_roi_html,
        )

    def shoreline_onclick_handler(
        self,
        event: str = None,
        id: int = None,
        properties: dict = None,
        **args,
    ):
        """On click handler for when unselected geojson is clicked.

        Adds object's id to selected_objects_set. Replaces current selected layer with a new one that includes
        recently clicked geojson.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked geojson. Defaults to None.
        """
        if properties is None:
            return
        # Add id of clicked shape to selected_set
        self.selected_shorelines_set.add(str(properties["id"]))
        # remove old selected layer
        self.remove_layer_by_name(SELECTED_LAYER_NAME)
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_shorelines_set, Shoreline.LAYER_NAME
            ),
            name=SELECTED_LAYER_NAME,
            hover_style={"fillColor": "orange", "fillOpacity": 0.1, "color": "orange"},
        )
        self.replace_layer_by_name(
            SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_shoreline_onclick_handler,
            on_hover=None,
        )

    def selected_shoreline_onclick_handler(
        self, event: str = None, id: str = None, properties: dict = None, **args
    ):
        """On click handler for selected geojson layer.

        Removes clicked layer's cid from the selected_set and replaces the select layer with a new one with
        the clicked layer removed from select_layer.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked selected geojson. Defaults to None.
        """
        if properties is None:
            return

        # Remove the current layers cid from selected set
        self.selected_shorelines_set.remove(str(properties["id"]))
        self.remove_layer_by_name(SELECTED_LAYER_NAME)
        # Recreate selected layers without layer that was removed
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_shorelines_set, Shoreline.LAYER_NAME
            ),
            name=SELECTED_LAYER_NAME,
            hover_style={"fillColor": "orange", "fillOpacity": 0.1, "color": "orange"},
        )
        self.replace_layer_by_name(
            SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_shoreline_onclick_handler,
            on_hover=None,
        )

    def selected_onclick_handler(
        self, event: str = None, id: str = None, properties: dict = None, **args
    ):
        """On click handler for selected geojson layer.

        Removes clicked layer's cid from the selected_set and replaces the select layer with a new one with
        the clicked layer removed from select_layer.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked selected geojson. Defaults to None.
        """
        if properties is None:
            return
        # Remove the current layers cid from selected set
        self.selected_set.remove(properties["id"])
        self.remove_layer_by_name(ROI.SELECTED_LAYER_NAME)
        # Recreate selected layers without layer that was removed
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_set, ROI.LAYER_NAME
            ),
            name=ROI.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        self.replace_layer_by_name(
            ROI.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.selected_onclick_handler,
            on_hover=self.update_roi_html,
        )

    def save_feature_to_file(
        self,
        feature: Union[Bounding_Box, Shoreline, Transects, ROI],
        feature_type: str = "",
    ):
        """
        Save a geographical feature (Bounding_Box, Shoreline, Transects, ROI) as a GeoJSON file.

        Args:
            feature (Union[Bounding_Box, Shoreline, Transects, ROI]): The geographical feature to save.
            feature_type (str, optional): The type of the feature, e.g. Bounding_Box, Shoreline, Transects, or ROI.
                                        Default value is an empty string.
        """
        exception_handler.can_feature_save_to_file(feature, feature_type)
        logger.info(f"Saving feature type({feature}) to file")
        if isinstance(feature, ROI):
            # raise exception if no rois were selected
            exception_handler.check_selected_set(self.selected_set)
            # save only the selected ROIs to file
            feature.gdf[feature.gdf["id"].isin(self.selected_set)].to_file(
                feature.filename, driver="GeoJSON"
            )
            print(f"Saved selected ROIs to {feature.filename}")
            logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")
        else:
            if hasattr(feature, "gdf"):
                feature.gdf.to_file(feature.filename, driver="GeoJSON")
                print(f"Save {feature.LAYER_NAME} to {feature.filename}")
                logger.info(f"Save {feature.LAYER_NAME} to {feature.filename}")
            else:
                logger.warning(f"Empty {feature.LAYER_NAME} cannot be saved to file")
                print(f"Empty {feature.LAYER_NAME} cannot be saved to file")

    def convert_selected_set_to_geojson(
        self, selected_set: set, layer_name: str, style: Optional[Dict] = None
    ) -> Dict:
        """Returns a geojson dict containing a FeatureCollection for all the geojson objects in the
        selected_set
        Args:
            selected_set (set): ids of selected geojson
            layer_name (str): name of the layer to get geometries from
            style (Optional[Dict]): style dictionary to be applied to each selected feature.
                If no style is provided then a default style is used:
                style = {
                    "color": "blue",
                    "weight": 2,
                    "fillColor": "blue",
                    "fillOpacity": 0.1,
                }
        Returns:
            Dict: geojson dict containing FeatureCollection for all geojson objects in selected_set
        """
        # create a new geojson dictionary to hold selected shapes
        if style is None:
            style = {
                "color": "blue",
                "weight": 2,
                "fillColor": "blue",
                "fillOpacity": 0.1,
            }
        selected_shapes = {"type": "FeatureCollection", "features": []}
        layer = self.map.find_layer(layer_name)
        # if ROI layer does not exist throw an error
        if layer is not None:
            exception_handler.check_empty_layer(layer, layer_name)
        # Copy only selected features with id in selected_set
        selected_features = [
            feature
            for feature in layer.data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        selected_shapes["features"] = [
            {**feature, "properties": {**feature["properties"], "style": style}}
            for feature in selected_features
        ]
        return selected_shapes
