import logging
from math import sqrt
from typing import Union, Optional, Callable, Dict
from enum import Enum

from coastseg.bbox import Bounding_Box
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI
from coastseg import exception_handler
from coastseg import coastseg_map
from geopandas import GeoDataFrame


# from geopandas import GeoDataFrame,GeoSeries
# from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

__all__ = ["Factory"]


def merge_rectangles(gdf: GeoDataFrame) -> GeoDataFrame:
    """
    Merges all rectangles in a GeoDataFrame into a single shape.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the rectangles.

    Returns:
        GeoDataFrame: A new GeoDataFrame containing a single shape that is the union of all rectangles in gdf.
                          The new GeoDataFrame has the same columns as the original.
    """
    # Ensure that the GeoDataFrame contains Polygons
    if not all(gdf.geometry.geom_type == "Polygon"):
        raise ValueError("All shapes in the GeoDataFrame must be Polygons.")

    # Merge all shapes into one
    merged_shape = gdf.unary_union

    # Create a new GeoDataFrame with the merged shape and the same columns as the original
    merged_gdf = GeoDataFrame([gdf.iloc[0]], geometry=[merged_shape], crs=gdf.crs)

    return merged_gdf


def create_shoreline(
    coastsegmap, gdf: Optional[GeoDataFrame] = None, **kwargs
) -> Shoreline:
    """
    Create a `Shoreline` object from a `GeoDataFrame` or a `coastsegmap`.

    Args:
        coastsegmap: A `coastsegmap` object.
        gdf: A `GeoDataFrame` of the shoreline, or `None` to load the shoreline from the `CoastlineMap`.
        **kwargs: Optional keyword arguments.

    Returns:
        A `Shoreline` object.

    Raises:
        Object_Not_Found: If the `coastsegmap` does not have a valid bounding box or shoreline.
    """
    if gdf is not None:
        shoreline = Shoreline(shoreline=gdf)
    else:
        # check if coastsegmap has a ROI
        if coastsegmap.rois is not None:
            if coastsegmap.rois.gdf.empty == False:
                # merge ROI geometeries together and use that as the bbbox
                merged_rois = merge_rectangles(coastsegmap.rois.gdf)
                shoreline = Shoreline(merged_rois)
                exception_handler.check_if_gdf_empty(shoreline.gdf, "shoreline")
        else:
            exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
            exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")
            shoreline = Shoreline(coastsegmap.bbox.gdf)
            exception_handler.check_if_gdf_empty(shoreline.gdf, "shoreline")

    logger.info("Shoreline were loaded on map")
    coastsegmap.shoreline = shoreline
    return shoreline


# def create_transects(
#     coastsegmap, gdf: Optional[GeoDataFrame] = None, **kwargs
# ) -> Transects:
#     if gdf is not None:
#         transects = Transects(transects=gdf)
#     else:
#         # check if coastsegmap has a ROI
#         if coastsegmap.rois is not None:
#             if coastsegmap.rois.gdf.empty == False:
#                 # merge ROI geometeries together and use that as the bbbox
#                 merged_rois = merge_rectangles(coastsegmap.rois.gdf)
#                 transects = Transects(merged_rois)
#                 exception_handler.check_if_gdf_empty(
#                     transects.gdf,
#                     "transects",
#                     "Transects Not Found in this region. Draw a new bounding box",
#                 )
#         else:
#             # otherwise check if coastsegmap has a bbox
#             exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
#             exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")

#             transects = Transects(coastsegmap.bbox.gdf)
#             exception_handler.check_if_gdf_empty(
#                 transects.gdf,
#                 "transects",
#                 "Transects Not Found in this region. Draw a new bounding box",
#             )

#     logger.info("Transects were loaded on map")
#     coastsegmap.transects = transects
#     return transects


def create_transects(
    coastsegmap, gdf: Optional[GeoDataFrame] = None, **kwargs
) -> Transects:
    if gdf is not None:
        transects = Transects(transects=gdf)
    else:
        # load the transects in all the ROIS in coastsegmap
        if coastsegmap.rois is not None:
            if coastsegmap.rois.gdf.empty == False:
                # merge ROI geometeries together and use that as the bbbox
                merged_rois = merge_rectangles(coastsegmap.rois.gdf)
                transects = Transects(merged_rois)
                exception_handler.check_if_gdf_empty(
                    transects.gdf,
                    "transects",
                    "Transects Not Found in this region. Draw a new bounding box",
                )
        else:
            # otherwise load the transects within a bbox in coastsegmap
            exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
            exception_handler.check_if_gdf_empty(coastsegmap.bbox.gdf, "bounding box")

            transects = Transects(coastsegmap.bbox.gdf)
            exception_handler.check_if_gdf_empty(
                transects.gdf,
                "transects",
                "Transects Not Found in this region. Draw a new bounding box",
            )

    logger.info("Transects were loaded on map")
    coastsegmap.transects = transects
    return transects


# def create_bbox(
#     coastsegmap, gdf: Optional[GeoDataFrame] = None, **kwargs
# ) -> Bounding_Box:
#     if gdf is not None:
#         bbox = Bounding_Box(gdf)
#         exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")
#     else:
#         geometry = coastsegmap.draw_control.last_draw["geometry"]
#         bbox = Bounding_Box(geometry)
#         exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")
#     coastsegmap.remove_bbox()
#     coastsegmap.draw_control.clear()
#     logger.info("Bounding Box was loaded on map")
#     coastsegmap.bbox = bbox
#     return bbox


def create_bbox(geometry: dict | GeoDataFrame, **kwargs) -> Bounding_Box:
    bbox = Bounding_Box(geometry)
    exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")
    return bbox


def create_rois(
    coastsegmap: coastseg_map, gdf: Optional[GeoDataFrame] = None, **kwargs
) -> ROI:
    if gdf is not None:
        rois = ROI(rois_gdf=gdf)
        exception_handler.check_if_gdf_empty(rois.gdf, "rois")
    else:
        # to create an ROI a bounding box must exist
        exception_handler.check_if_None(coastsegmap.bbox, "bounding box")
        # generate a shoreline within the bounding box
        if coastsegmap.shoreline is None:
            coastsegmap.load_feature_on_map("shoreline")
        logger.info(
            f"coastsegmap.shoreline:{coastsegmap.shoreline}\ncoastsegmap.bbox:{coastsegmap.bbox}"
        )

        lg_area = kwargs.get("lg_area")
        sm_area = kwargs.get("sm_area")
        units = kwargs.get("units")
        if lg_area is None or sm_area is None or units is None:
            raise Exception("Must provide ROI area and units")

        # if units is kilometers convert to meters
        if units == "km²":
            sm_area = sm_area * (10**6)
            lg_area = lg_area * (10**6)

        # get length of ROI square by taking square root of area
        small_len = sqrt(sm_area)
        large_len = sqrt(lg_area)

        # create rois within the bbox that intersect shorelines
        rois = ROI(
            coastsegmap.bbox.gdf,
            coastsegmap.shoreline.gdf,
            square_len_lg=large_len,
            square_len_sm=small_len,
        )
        exception_handler.check_if_gdf_empty(rois.gdf, "rois")

    coastsegmap.remove_all_rois()
    logger.info("ROIs were loaded on map")
    coastsegmap.rois = rois
    return rois


class FeatureType(Enum):
    SHORELINE = "shoreline"
    TRANSECTS = "transects"
    BBOX = "bbox"
    ROIS = "rois"


class Factory:
    # this dictionarymaps the feature name to the function that creates it
    # ex. "shoreline" maps to create_shoreline function
    _feature_makers: Dict[str, Callable] = {
        "shoreline": create_shoreline,
        "shorelines": create_shoreline,
        "transects": create_transects,
        "transect": create_transects,
        "bbox": create_bbox,
        "rois": create_rois,
        "roi": create_rois,
    }

    # @staticmethod
    # def make_feature(
    #     coastsegmap: "CoastSeg_Map",
    #     feature_name: str,
    #     gdf: Optional[GeoDataFrame] = None,
    #     **kwargs,
    # ) -> Union[Shoreline, Transects, Bounding_Box, ROI]:
    #     logger.info(
    #         f"feature_name {feature_name}\ncoastsegmap: {coastsegmap}\nGdf: {gdf}\nkwargs: {kwargs}"
    #     )
    #     # get the function that can be used to create the feature_name
    #     # ex. "shoreline" would get the create_shoreline function
    #     feature_maker = Factory._feature_makers.get(feature_name.lower())
    #     # if a geodataframe is provided
    #     if gdf is not None:
    #         if gdf.empty:
    #             return None

    #     return feature_maker(coastsegmap, gdf, **kwargs)
    @staticmethod
    def make_feature(
        feature_name: str,
        geometry: dict | GeoDataFrame = None,
        **kwargs,
    ) -> Union[Shoreline, Transects, Bounding_Box, ROI]:
        logger.info(
            f"feature_name {feature_name}\ngeometry: {geometry}\nkwargs: {kwargs}"
        )
        # get the function that can be used to create the feature_name
        # ex. "shoreline" would get the create_shoreline function
        feature_maker = Factory._feature_makers.get(feature_name.lower())
        # If an empty geodataframe is provided return None because nothing can be created
        if not geometry:
            return None
        elif isinstance(geometry, GeoDataFrame):
            if geometry.empty:
                return None
        elif isinstance(geometry, dict):
            if geometry == {}:
                return None
        # return the new feature
        return feature_maker(geometry, **kwargs)
