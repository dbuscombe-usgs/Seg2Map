# Standard library imports
import logging
from typing import Union, List
import datetime

# Internal dependencies imports
from coastseg import common
from coastseg.common import validate_geometry_types
from coastseg import exceptions

# External dependencies imports
import geopandas as gpd
import pandas as pd
from shapely import geometry
from ipyleaflet import GeoJSON

# from coastseg.extracted_shoreline import Extracted_Shoreline

logger = logging.getLogger(__name__)

__all__ = ["ROI"]


class ROI:
    """A class that controls all the ROIs on the map"""

    LAYER_NAME = "ROIs"
    SELECTED_LAYER_NAME = "Selected ROIs"
    MAX_SIZE = 98000000  # 98km^2 area
    MIN_SIZE = 0

    def __init__(
        self,
        bbox: gpd.GeoDataFrame = None,
        shoreline: gpd.GeoDataFrame = None,
        rois_gdf: gpd.GeoDataFrame = None,
        square_len_lg: float = 0,
        square_len_sm: float = 0,
        filename: str = None,
    ):
        # gdf : geodataframe of ROIs
        self.gdf = gpd.GeoDataFrame()
        # roi_settings : after ROIs have been downloaded holds all download settings
        self.roi_settings = {}
        # extract_shorelines : dictionary with ROIs' ids as the keys holding the extracted shorelines
        # ex. {'1': Extracted Shoreline()}
        self.extracted_shorelines = {}
        # cross_shore_distancess : dictionary with of cross-shore distance along each of the transects. Not tidally corrected.
        self.cross_shore_distances = {}
        self.filename = filename or "rois.geojson"

        if rois_gdf is not None:
            self._initialize_from_roi_gdf(rois_gdf)
        else:
            self._initialize_from_bbox_and_shoreline(
                bbox, shoreline, square_len_lg, square_len_sm
            )

    def __str__(self):
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 5 rows as a string
        first_rows = self.gdf.head().to_string()
        # Get CRS information
        crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
        extracted_shoreline_info = ""
        for key in self.extracted_shorelines.keys():
            if hasattr(self.extracted_shorelines[key], "gdf") and (
                isinstance(self.extracted_shorelines[key].gdf, gpd.GeoDataFrame)
            ):
                if not self.extracted_shorelines[key].gdf.empty:
                    extracted_shoreline_info.join(
                        f"ROI ID {key}:\n{len(self.extracted_shorelines[key].gdf)}\n"
                    )
        return f"ROI:\nROI IDs: {self.get_ids()}\nROI IDs with extracted shorelines: {extracted_shoreline_info}\nROI IDs with shoreline transect intersections: {list(self.cross_shore_distances.keys())}\n gdf:\n{crs_info}\nColumns and Data Types:\n{col_info}\n\nFirst 5 Rows:\n{first_rows}"

    def __repr__(self):
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 5 rows as a string
        first_rows = self.gdf.head().to_string()
        # Get CRS information
        crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
        extracted_shoreline_info = ""
        for key in self.extracted_shorelines.keys():
            if hasattr(self.extracted_shorelines[key], "gdf") and (
                isinstance(self.extracted_shorelines[key].gdf, gpd.GeoDataFrame)
            ):
                if not self.extracted_shorelines[key].gdf.empty:
                    extracted_shoreline_info.join(
                        f"ROI ID {key}:\n{len(self.extracted_shorelines[key].gdf)}\n"
                    )
        return f"ROI:\nROI IDs: {self.get_ids()}\nROI IDs with extracted shorelines: {extracted_shoreline_info}\nROI IDs with shoreline transect intersections: {list(self.cross_shore_distances.keys())}\n gdf:\n{crs_info}\nColumns and Data Types:\n{col_info}\n\nFirst 5 Rows:\n{first_rows}"

    def remove_by_id(
        self, ids_to_drop: list | set | tuple | str | int
    ) -> gpd.GeoDataFrame:
        if self.gdf.empty or "id" not in self.gdf.columns or ids_to_drop is None:
            return self.gdf
        if isinstance(ids_to_drop, (str, int)):
            ids_to_drop = [
                str(ids_to_drop)
            ]  # Convert to list and ensure ids are strings
        # Ensure all elements in ids_to_drop are strings for consistent comparison
        ids_to_drop = set(map(str, ids_to_drop))
        logger.info(f"ids_to_drop from roi: {ids_to_drop}")
        # drop the ids from the geodataframe
        self.gdf = self.gdf[~self.gdf["id"].astype(str).isin(ids_to_drop)]
        # remove the corresponding extracted shorelines
        for roi_id in ids_to_drop:
            self.remove_extracted_shorelines(roi_id)

        return self.gdf

    def _initialize_from_roi_gdf(self, rois_gdf: gpd.GeoDataFrame) -> None:
        """
        Initialize the `gdf` attribute from a GeoDataFrame of ROIs.

        Args:
            rois_gdf: A GeoDataFrame of ROIs.

        Returns:
            None.

        Raises:
            None.
        """
        # make sure to perform a CRS check here too

        rois_gdf = common.preprocess_geodataframe(
            rois_gdf,
            columns_to_keep=["id", "geometry"],
            create_ids=True,
            output_crs="EPSG:4326",
        )
        validate_geometry_types(
            rois_gdf, set(["Polygon", "MultiPolygon"]), feature_type="ROI"
        )
        # make sure all the ids  are unique
        rois_gdf = common.create_unique_ids(rois_gdf, prefix_length=3)
        # get row ids of ROIs with area that's too large
        drop_ids = common.get_ids_with_invalid_area(
            rois_gdf, max_area=ROI.MAX_SIZE, min_area=ROI.MIN_SIZE
        )
        if drop_ids:
            logger.info(f"Dropping ROIs that are an invalid size {drop_ids}")
            rois_gdf.drop(index=drop_ids, axis=0, inplace=True)
            if rois_gdf.empty:
                raise exceptions.InvalidSize(
                    "The ROI(s) had an invalid size.",
                    "ROI",
                    max_size=ROI.MAX_SIZE,
                    min_size=ROI.MIN_SIZE,
                )

        self.gdf = rois_gdf

    def _initialize_from_bbox_and_shoreline(
        self,
        bbox: gpd.GeoDataFrame,
        shoreline: gpd.GeoDataFrame,
        square_len_lg: float,
        square_len_sm: float,
    ) -> None:
        """
        Initialize the `gdf` attribute from a bounding box and shoreline.

        Args:
            bbox: A GeoDataFrame representing the bounding box.
            shoreline: A GeoDataFrame representing the shoreline.
            square_len_lg: The length of the larger square.
            square_len_sm: The length of the smaller square.

        Returns:
            None.

        Raises:
            Object_Not_Found: If the `shoreline` or `bbox` is `None` or empty.
            ValueError: If `square_len_sm` or `square_len_lg` is not greater than 0.
        """
        if shoreline is None:
            raise exceptions.Object_Not_Found("shorelines")
        if bbox is None:
            raise exceptions.Object_Not_Found("bounding box")
        if shoreline.empty:
            raise exceptions.Object_Not_Found("shorelines")
        if bbox.empty:
            raise exceptions.Object_Not_Found("bounding box")

        if square_len_sm == square_len_lg == 0:
            logger.error("Invalid square size for ROI")
            raise ValueError("Invalid square size for ROI. Must be greater than 0")

        self.gdf = self.create_geodataframe(
            bbox, shoreline, square_len_lg, square_len_sm
        )

    def get_roi_settings(self, roi_id: str = "") -> dict:
        """
        Retrieve the settings for a specific ROI or all ROI settings.

        Args:
            roi_id (str, optional): The ID of the ROI to retrieve settings for. 
                If not provided, all ROI settings will be returned. Defaults to "".

        Returns:
            dict: The settings for the specified ROI, or all ROI settings if no ROI ID is provided.
        """
        if not hasattr(self, "roi_settings"):
            self.roi_settings = {}
        if roi_id is None:
            return {}
        if not isinstance(roi_id, str):
            raise TypeError("roi_id must be a string")
        
        if roi_id in self.roi_settings:
            logger.info(f"self.roi_settings[roi_id]: {self.roi_settings[roi_id]}")
            return self.roi_settings[roi_id]
        else:
            if roi_id == "":
                    logger.info(f"self.roi_settings: {self.roi_settings}")
                    return self.roi_settings
            else:
                return {}

    def set_roi_settings(self, roi_settings: dict) -> None:
        """Sets the ROI settings dictionary to the specified value.

        Args:
            roi_settings (dict): A dictionary of settings for the ROI.
        """
        if roi_settings is None:
            raise ValueError("roi_settings cannot be None")
        if not isinstance(roi_settings, dict):
            raise TypeError("roi_settings must be a dictionary")

        logger.info(f"Saving roi_settings {roi_settings}")
        self.roi_settings = roi_settings

    def update_roi_settings(self, new_settings: dict) -> None:
        """Updates the ROI settings dictionary with the specified values.

        Args:
            new_settings (dict): A dictionary of new settings for the ROI.
        """
        if new_settings is None:
            raise ValueError("new_settings cannot be None")

        logger.info(f"Updating roi_settings with {new_settings}")
        if self.roi_settings is None:
            self.roi_settings = new_settings
        else:
            self.roi_settings.update(new_settings)
        return self.roi_settings

    def get_extracted_shoreline(self, roi_id: str) -> Union[None, dict]:
        """Returns the extracted shoreline for the specified ROI ID.

        Args:
            roi_id (str): The ID of the ROI to retrieve the shoreline for.

        Returns:
            Union[None, dict]: The extracted shoreline dictionary for the specified ROI ID, or None if it does not exist.
        """
        return self.extracted_shorelines.get(roi_id)

    def get_ids(self) -> list:
        """Returns list of all roi ids"""
        if "id" not in self.gdf.columns:
            return []
        return self.gdf["id"].tolist()

    def get_ids_with_extracted_shorelines(self) -> Union[None, List[str]]:
        """Returns list of roi ids that had extracted shorelines"""
        return list(self.get_all_extracted_shorelines().keys())

    def add_geodataframe(self, gdf: gpd.GeoDataFrame) -> "ROI":
        """Adds the geodataframe to the map"""
        # check if geodataframe column has 'id' column and add one if one doesn't exist
        if "id" not in gdf.columns:
            gdf["id"] = gdf.index.astype(str).tolist()
        # get row ids of ROIs with area that's too large
        drop_ids = common.get_ids_with_invalid_area(gdf)
        if drop_ids:
            print("Dropping ROIs that are an invalid size ")
            logger.info(f"Dropping ROIs that are an invalid size {drop_ids}")
            gdf.drop(index=drop_ids, axis=0, inplace=True)
        # Combine the two GeoDataFrames and drop duplicates from columns "id" and "geometry"
        combined_gdf = pd.concat([self.gdf, gdf], axis=0).drop_duplicates(
            subset=["id", "geometry"]
        )
        # Convert the combined DataFrame back to a GeoDataFrame
        self.gdf = gpd.GeoDataFrame(combined_gdf, crs=self.gdf.crs)
        return self

    def get_all_extracted_shorelines(self) -> dict:
        """Returns a dictionary of all extracted shorelines.

        Returns:
            dict: A dictionary containing all extracted shorelines, indexed by ROI ID.
        """
        if not hasattr(self, "extracted_shorelines"):
            self.extracted_shorelines = {}
        return self.extracted_shorelines

    def remove_extracted_shorelines(
        self, roi_id: str = None, remove_all: bool = False
    ) -> None:
        """Removes the extracted shoreline for the specified ROI ID, or all extracted shorelines.

        Args:
            roi_id (str, optional): The ID of the ROI to remove the shoreline for. Defaults to None.
            remove_all (bool, optional): Whether to remove all extracted shorelines. Defaults to False.
        """
        if roi_id in self.extracted_shorelines:
            del self.extracted_shorelines[roi_id]
        if remove_all:
            del self.extracted_shorelines
            self.extracted_shorelines = {}

    def remove_selected_shorelines(
        self, roi_id: str, dates: list[datetime.datetime], satellites: list[str]
    ) -> None:
        """
        Removes selected shorelines for a specific region of interest (ROI).

        Args:
            roi_id (str): The ID of the ROI.
            dates (list[datetime.datetime]): A list of dates for which the shorelines should be removed.
            satellites (list[str]): A list of satellite names for which the shorelines should be removed.

        Returns:
            None
        """
        if roi_id in self.get_ids_with_extracted_shorelines():
            extracted_shoreline = self.get_extracted_shoreline(roi_id)
            if extracted_shoreline is not None:
                extracted_shoreline.remove_selected_shorelines(dates, satellites)

    def add_extracted_shoreline(
        self,
        extracted_shoreline: "coastseg.extracted_shoreline.Extracted_Shoreline",
        roi_id: str,
    ) -> None:
        """Adds an extracted shoreline dictionary to the collection, indexed by the specified ROI ID.

        Args:
            extracted_shoreline (Extracted_Shoreline): The extracted shoreline dictionary to add.
            roi_id (str): The ID of the ROI to associate the shoreline with.
        """
        self.extracted_shorelines[roi_id] = extracted_shoreline
        logger.info(f"New extracted shoreline added for ROI {roi_id}")
        # logger.info(f"New extracted shoreline added for ROI {roi_id}: {self.extracted_shorelines}")

    def get_cross_shore_distances(self, roi_id: str) -> Union[None, dict]:
        """Returns the cross shore distance for the specified ROI ID.

        Args:
            roi_id (str): The ID of the ROI to retrieve the shoreline for.

        Returns:
            Union[None, dict]: Thecross shore distance dictionary for the specified ROI ID, or None if it does not exist.
        """
        self.cross_shore_distances.get(roi_id, {})
        if self.cross_shore_distances.get(roi_id, {}) == {}:
            logger.info(f"ROI: {roi_id} has no cross shore distance")
        else:
            logger.info(
                f"ROI: {roi_id} cross distance with keys : {self.cross_shore_distances.get(roi_id,{})}"
            )
        return self.cross_shore_distances.get(roi_id, 0)

    def add_cross_shore_distances(
        self, cross_shore_distance: dict, roi_id: str
    ) -> None:
        """Adds an cross_shore_distance dictionary to the collection, indexed by the specified ROI ID.

        Args:
            cross_shore_distance (dict): The cross_shore_distance dictionary to add.
            roi_id (str): The ID of the ROI to associate the cross_shore_distance dictionary
        """
        self.cross_shore_distances[roi_id] = cross_shore_distance
        # logger.info(f"Newly added cross_shore_distance: {cross_shore_distance}")

    def get_all_cross_shore_distances(
        self,
    ) -> None:
        """Returns a dictionary of all cross shore distances

        Returns:
            dict: A dictionary containing all cross shore distances, indexed by ROI ID.
        """
        return self.cross_shore_distances

    def remove_cross_shore_distance(
        self, roi_id: str = None, remove_all: bool = False
    ) -> None:
        """Removes the cross shore distance for the specified ROI ID, or all  cross shore distances.

        Args:
            roi_id (str, optional): The ID of the ROI to remove the shoreline for. Defaults to None.
            remove_all (bool, optional): Whether to remove all  cross shore distances. Defaults to False.
        """
        if roi_id in self.cross_shore_distances:
            del self.cross_shore_distances[roi_id]
        if remove_all:
            self.cross_shore_distances = {}

    def create_geodataframe(
        self,
        bbox: gpd.geodataframe,
        shoreline: gpd.GeoDataFrame,
        large_length: float = 7500,
        small_length: float = 5000,
        crs: str = "EPSG:4326",
    ) -> gpd.GeoDataFrame:
        """Generates series of overlapping ROIS along shoreline on map using fishnet method
        with the crs specified by crs
        Args:

            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: a series of ROIs along the shoreline. The ROIs are squares with side lengths of both
            large_length and small_length
        """
        # Create a single set of fishnets with square size = small and/or large side lengths that overlap each other
        # logger.info(f"Small Length: {small_length}  Large Length: {large_length}")
        if small_length == 0 or large_length == 0:
            # logger.info("Creating one fishnet")
            # create a fishnet geodataframe with square size of either large_length or small_length
            fishnet_size = large_length if large_length != 0 else small_length
            fishnet_intersect_gdf = self.get_fishnet_gdf(bbox, shoreline, fishnet_size)
        else:
            # logger.info("Creating two fishnets")
            # Create two fishnets, one big (2000m) and one small(1500m) so they overlap each other
            fishnet_gpd_large = self.get_fishnet_gdf(bbox, shoreline, large_length)
            fishnet_gpd_small = self.get_fishnet_gdf(bbox, shoreline, small_length)
            # logger.info(f"fishnet_gpd_large : {fishnet_gpd_large}")
            # logger.info(f"fishnet_gpd_small : {fishnet_gpd_small}")
            # Concat the fishnets together to create one overlapping set of rois
            fishnet_intersect_gdf = gpd.GeoDataFrame(
                pd.concat([fishnet_gpd_large, fishnet_gpd_small], ignore_index=True)
            )

        # clean the geodataframe
        fishnet_intersect_gdf = common.preprocess_geodataframe(
            fishnet_intersect_gdf,
            columns_to_keep=[
                "id",
                "geometry",
            ],
            create_ids=True,
        )
        # make sure all the ids are unique
        fishnet_intersect_gdf = common.create_unique_ids(
            fishnet_intersect_gdf, prefix_length=3
        )
        # logger.info(f"Created fishnet_intersect_gdf: {fishnet_intersect_gdf}")
        return fishnet_intersect_gdf

    def style_layer(self, geojson: dict, layer_name: str) -> GeoJSON:
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer
        Returns:
            "ipyleaflet.GeoJSON": ROIs as GeoJson layer styled with yellow dashes
        """
        assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style={
                "color": "#555555",
                "fill_color": "#555555",
                "fillOpacity": 0.1,
                "weight": 1,
            },
            hover_style={"color": "red", "fillOpacity": 0.1, "color": "crimson"},
        )

    def fishnet_intersection(
        self, fishnet: gpd.GeoDataFrame, data: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Returns the intersection of a fishnet grid with given data.

        Args:
            fishnet (gpd.GeoDataFrame): GeoDataFrame consisting of equal-sized squares (fishnet grid).
            data (gpd.GeoDataFrame): GeoDataFrame containing vector or polygon data to intersect with the fishnet.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame representing the intersection of the fishnet and the input data.
        """
        # Perform a spatial join between the fishnet and data to find the intersecting geometries
        intersection_gdf = gpd.sjoin(
            left_df=fishnet, right_df=data, how="inner", predicate="intersects"
        )

        # Remove unnecessary columns, keeping only the geometry column
        columns_to_keep = ["geometry"]
        intersection_gdf = intersection_gdf[columns_to_keep]

        # Remove duplicate geometries
        intersection_gdf.drop_duplicates(
            keep="first", subset=["geometry"], inplace=True
        )

        return intersection_gdf

    def create_rois(
        self,
        bbox: gpd.GeoDataFrame,
        square_size: float,
        input_espg="epsg:4326",
        output_epsg="epsg:4326",
    ) -> gpd.geodataframe:
        """Creates a fishnet of square shaped ROIs with side length= square_size

        Args:
            bbox (geopandas.geodataframe.GeoDataFrame): bounding box(bbox) where the ROIs will be generated
            square_size (float): side length of square ROI in meters
            input_espg (str, optional): espg code bbox is currently in. Defaults to "epsg:4326".
            output_epsg (str, optional): espg code the ROIs will output to. Defaults to "epsg:4326".

        Returns:
            gpd.geodataframe: geodataframe containing all the ROIs
        """
        projected_espg = common.get_epsg_from_geometry(bbox.iloc[0]["geometry"])
        logger.info(f"ROI: projected_espg_code: {projected_espg}")
        # project geodataframe to new CRS specified by utm_code
        projected_bbox_gdf = bbox.to_crs(projected_espg)
        # create fishnet of rois
        fishnet = self.create_fishnet(
            projected_bbox_gdf, projected_espg, output_epsg, square_size
        )
        return fishnet

    def create_fishnet(
        self,
        bbox_gdf: gpd.GeoDataFrame,
        input_espg: str,
        output_epsg: str,
        square_size: float = 1000,
    ) -> gpd.geodataframe:
        """Returns a fishnet of ROIs that intersects the bounding box specified by bbox_gdf where each ROI(square) has a side length = square size(meters)

        Args:
            bbox_gdf (gpd.geodataframe): Bounding box that fishnet intersects.
            input_espg (str): espg string that bbox_gdf is projected in
            output_epsg (str): espg to convert the fishnet of ROIs to.
            square_size (int, optional): Size of each square in fishnet(meters). Defaults to 1000.

        Returns:
            gpd.geodataframe: fishnet of ROIs that intersects bbox_gdf. Each ROI has a side lenth = sqaure_size
        """
        minX, minY, maxX, maxY = bbox_gdf.total_bounds
        # Create a fishnet where each square has side length = square size
        x, y = (minX, minY)
        geom_array = []
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon(
                    [
                        (x, y),
                        (x, y + square_size),
                        (x + square_size, y + square_size),
                        (x + square_size, y),
                        (x, y),
                    ]
                )
                # add each square to geom_array
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size

        # create geodataframe to hold all the (rois)squares
        fishnet = gpd.GeoDataFrame(geom_array, columns=["geometry"]).set_crs(input_espg)
        logger.info(
            f"\n ROIs area before conversion to {output_epsg}:\n {fishnet.area} for CRS: {input_espg}"
        )
        fishnet = fishnet.to_crs(output_epsg)
        return fishnet

    def get_fishnet_gdf(
        self,
        bbox_gdf: gpd.GeoDataFrame,
        shoreline_gdf: gpd.GeoDataFrame,
        square_length: int = 1000,
    ) -> gpd.GeoDataFrame:
        """
        Returns fishnet where it intersects the shoreline

        Args:
            bbox_gdf (GeoDataFrame): bounding box (bbox) around shoreline
            shoreline_gdf (GeoDataFrame): shoreline in the bbox
            square_size (int, optional): size of each square in the fishnet. Defaults to 1000.

        Returns:
            GeoDataFrame: intersection of shoreline_gdf and fishnet. Only squares that intersect shoreline are kept
        """
        # Get the geodataframe for the fishnet within the bbox
        fishnet = self.create_rois(bbox_gdf, square_length)
        # Get the geodataframe for the fishnet intersecting the shoreline
        fishnet_intersection = self.fishnet_intersection(fishnet, shoreline_gdf)
        return fishnet_intersection
