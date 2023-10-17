# Standard library imports
import logging
import os
from typing import List, Optional

# Internal dependencies imports
from coastseg.common import (
    preprocess_geodataframe,
    create_unique_ids,
    validate_geometry_types,
)

# External dependencies imports
import geopandas as gpd
import pandas as pd
from ipyleaflet import GeoJSON


logger = logging.getLogger(__name__)


def load_intersecting_transects(
    rectangle: gpd.GeoDataFrame,
    transect_files: List[str],
    transect_dir: str,
    columns_to_keep: set = set(["id", "geometry", "slope"]),
) -> gpd.GeoDataFrame:
    """
    Loads transects from a list of GeoJSON files in the transect directory, selects the transects that intersect with
    a rectangle defined by a GeoDataFrame, and returns a new GeoDataFrame with the selected columns ('id', 'geometry', 'slope').

    Args:
        rectangle (gpd.GeoDataFrame): A GeoDataFrame defining the rectangle to select transects within.
        transect_files (List[str]): A list of filenames of the GeoJSON transect files to load.
        transect_dir (str): The directory where the GeoJSON transect files are located.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with the selected columns ('id', 'geometry', 'slope') containing the transects
        that intersect with the rectangle.
    """
    # Create an empty GeoDataFrame to hold the selected transects
    selected_transects = gpd.GeoDataFrame(columns=list(columns_to_keep))

    # Get the bounding box of the rectangle
    bbox = rectangle.bounds.iloc[0].tolist()

    # Iterate over each transect file and select the transects that intersect with the rectangle
    for transect_file in transect_files:
        transects_name = os.path.splitext(transect_file)[0]
        transect_path = os.path.join(transect_dir, transect_file)
        transects = gpd.read_file(transect_path, bbox=bbox)
        if transects.empty:
            logger.info("Skipping %s", transects_name)
            continue
        elif not transects.empty:
            logger.info("Adding transects from %s", transects_name)
            transects = preprocess_geodataframe(
                transects, columns_to_keep=list(columns_to_keep), create_ids=False
            )
            # Append the selected transects to the output GeoDataFrame
            selected_transects = pd.concat(
                [selected_transects, transects], ignore_index=True
            )
    selected_transects = preprocess_geodataframe(
        selected_transects, columns_to_keep=list(columns_to_keep), create_ids=True
    )
    validate_geometry_types(
        selected_transects,
        set(["LineString", "MultiLineString"]),
        feature_type="transects",
    )
    # make sure all the ids in selected_transects are unique
    selected_transects = create_unique_ids(selected_transects, prefix_length=3)
    return selected_transects


class Transects:
    """A class representing a collection of transects within a specified bounding box."""

    LAYER_NAME = "transects"
    COLUMNS_TO_KEEP = set(
        [
            "id",
            "geometry",
            "slope",
            "distance",
            "feature_x",
            "feature_y",
            "nearest_x",
            "nearest_y",
        ]
    )

    # Define columns of interest and their descriptions:
    # id: unique identifier for each transect
    # geometry: the geometric shape, position, and configuration of the transect
    # slope: represents the beach face slope, used for tidal correction of transect-based data
    # distance: represents the distance in degrees between the slope datum location and the transect location
    # feature_x: x-coordinate of the transect location
    # feature_y: y-coordinate of the transect location
    # nearest_x: x-coordinate of the nearest slope location to the transect
    # nearest_y: y-coordinate of the nearest slope location to the transect

    def __init__(
        self,
        bbox: gpd.GeoDataFrame = None,
        transects: gpd.GeoDataFrame = None,
        filename: str = None,
    ):
        """
        Initialize a Transects object with either a bounding box GeoDataFrame, a transects GeoDataFrame,
        or a filename string.
        """
        self.gdf = gpd.GeoDataFrame()
        self.filename = filename if filename else "transects.geojson"
        self.initialize_transects(bbox, transects)

    def __str__(self):
        return f"Transects: geodataframe {self.gdf}"

    def __repr__(self):
        return f"Transects: geodataframe {self.gdf}"

    def initialize_transects(
        self,
        bbox: Optional[gpd.GeoDataFrame] = None,
        transects: Optional[gpd.GeoDataFrame] = None,
    ):
        if transects is not None:
            if isinstance(transects, gpd.GeoDataFrame) and not transects.empty:
                self.initialize_transects_with_transects(transects)
        elif bbox is not None:
            if isinstance(bbox, gpd.GeoDataFrame) and not bbox.empty:
                self.initialize_transects_with_bbox(bbox)
        else:
            # if the geodataframe was empty, None or the wrong type return None
            return None

    def initialize_transects_with_transects(self, transects: gpd.GeoDataFrame):
        """
        Initialize transects with the provided transects in a GeoDataFrame.
        """
        if not transects.empty:
            if not transects.crs:
                logger.warning(
                    f"transects did not have a crs converting to crs 4326 \n {transects}"
                )
                transects.set_crs("EPSG:4326", inplace=True)
            transects = preprocess_geodataframe(
                transects,
                columns_to_keep=list(Transects.COLUMNS_TO_KEEP),
                create_ids=True,
                output_crs="EPSG:4326",
            )
            validate_geometry_types(
                transects,
                set(["LineString", "MultiLineString"]),
                feature_type="transects",
                help_message=f"The uploaded transects need to be LineStrings.",
            )
            # if not all the ids in transects are unique then create unique ids
            transects = create_unique_ids(transects, prefix_length=3)
            # @todo add the transects to the current dataframe
            # @todo make sure none of the ids already exist in the dataframe. this can be a flag to turn an exception on/off
            self.gdf = transects

    def initialize_transects_with_bbox(self, bbox: gpd.GeoDataFrame):
        """
        Load transects within the bounding box. The transects will NOT be clipped to the bounding box.
        Args:
            bbox (gpd.GeoDataFrame): bounding box
        """
        if not bbox.empty:
            self.gdf = self.create_geodataframe(bbox)

    def create_geodataframe(
        self,
        bbox: gpd.GeoDataFrame,
        crs: str = "EPSG:4326",
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by crs
        Args:
             bbox (gpd.GeoDataFrame): Bounding box being searched for transects
             crs (str, optional): Coordinate reference system string. Defaults to 'EPSG:4326'.
        Returns:
            gpd.GeoDataFrame: geodataframe with geometry column = bbox and given crs
        """
        # create a new dataframe that only contains the geometry column of the bbox
        bbox = bbox[["geometry"]]
        # get transect geosjson files that intersect with bounding box
        intersecting_transect_files = self.get_intersecting_files(bbox)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        transect_dir = os.path.abspath(os.path.join(script_dir, "transects"))
        # for each transect file clip it to the bbox and add to map
        transects_in_bbox = load_intersecting_transects(
            bbox,
            intersecting_transect_files,
            transect_dir,
            columns_to_keep=list(Transects.COLUMNS_TO_KEEP),
        )
        if transects_in_bbox.empty:
            logger.warning("No transects found here.")
        # remove z-axis from transects
        transects_in_bbox = preprocess_geodataframe(
            transects_in_bbox,
            columns_to_keep=list(Transects.COLUMNS_TO_KEEP),
            create_ids=True,
        )
        validate_geometry_types(
            transects_in_bbox,
            set(["LineString", "MultiLineString"]),
            feature_type="transects",
        )
        # make sure all the ids in transects_in_bbox are unique
        transects_in_bbox = create_unique_ids(transects_in_bbox, prefix_length=3)

        if not transects_in_bbox.empty:
            transects_in_bbox.to_crs(crs, inplace=True)

        return transects_in_bbox

    def style_layer(self, geojson: dict, layer_name: str) -> dict:
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer

        Returns:
            "ipyleaflet.GeoJSON": transects as styled GeoJSON layer
        """
        assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        # Add style to each feature in the geojson
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style={
                "color": "grey",
                "fill_color": "grey",
                "opacity": 1,
                "fillOpacity": 0.2,
                "weight": 2,
            },
            hover_style={"color": "blue", "fillOpacity": 0.7},
        )

    def get_intersecting_files(self, bbox_gdf: gpd.GeoDataFrame) -> list:
        """Returns a list of filenames that intersect with bbox_gdf

        Args:
            gpd_bbox (geopandas.geodataframe.GeoDataFrame): bbox containing ROIs
            type (str): to be used later

        Returns:
            list: intersecting_files containing filenames whose contents intersect with bbox_gdf
        """
        # dataframe containing total bounding box for each transects file
        total_bounds_df = self.load_total_bounds_df()
        # filenames where transects/shoreline's bbox intersect bounding box drawn by user
        intersecting_files = []
        for filename in total_bounds_df.index:
            minx, miny, maxx, maxy = total_bounds_df.loc[filename]
            intersection_df = bbox_gdf.cx[minx:maxx, miny:maxy]
            # save filenames where gpd_bbox & bounding box for set of transects intersect
            if intersection_df.empty == False:
                intersecting_files.append(filename)
        return intersecting_files

    def load_total_bounds_df(self) -> pd.DataFrame:
        """Returns dataframe containing total bounds for each set of shorelines in the csv file specified by location

        Args:
            location (str, optional): determines whether usa or world shoreline bounding boxes are loaded. Defaults to 'usa'.
            can be either 'world' or 'usa'

        Returns:
            pd.DataFrame:  Returns dataframe containing total bounds for each set of shorelines
        """
        # Load in the total bounding box from csv
        # Create the directory to hold the downloaded shorelines from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bounding_box_dir = os.path.abspath(os.path.join(script_dir, "bounding_boxes"))
        if not os.path.exists(bounding_box_dir):
            os.mkdir(bounding_box_dir)

        transects_csv = os.path.join(bounding_box_dir, "transects_bounding_boxes.csv")
        if not os.path.exists(transects_csv):
            print("Did not find transects csv at ", transects_csv)
        else:
            total_bounds_df = pd.read_csv(transects_csv)

        total_bounds_df.index = total_bounds_df["filename"]
        if "filename" in total_bounds_df.columns:
            total_bounds_df.drop("filename", axis=1, inplace=True)
        return total_bounds_df
