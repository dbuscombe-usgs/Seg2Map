# Standard library imports
import os
import logging
import json
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union
import traceback

# Third-party imports
import pandas as pd
import geopandas as gpd

from ipyleaflet import DrawControl, LayersControl, WidgetControl, GeoJSON
from leafmap import Map
from ipywidgets import Layout, HTML, HBox
from tqdm.auto import tqdm
import traitlets
from shapely.geometry import Polygon
from typing import Dict, Optional, Set

# Internal/Local imports: specific classes/functions
from coastseg.bbox import Bounding_Box
from coastseg.shoreline import Shoreline
from coastseg.transects import Transects
from coastseg.roi import ROI
from coastseg.factory import Factory


# Internal/Local imports: modules
from coastseg import (
    common,
    exceptions,
    exception_handler,
)

logger = logging.getLogger(__name__)

# global variables
SELECTED_LAYER_NAME = "Selected Shorelines"


# Helper functions


def style_layer(
    geojson: Dict,
    layer_name: str,
    style: Dict[str, Union[str, float]] = None,
    hover_style: Dict[str, Union[str, float]] = {},
) -> GeoJSON:
    """Return styled GeoJson object with layer name

    Args:
        geojson (dict): geojson dictionary to be styled
        layer_name(str): name of the GeoJSON layer
        style (dict): styling attributes for the GeoJSON
        hover_style (dict): hover styling attributes for the GeoJSON

    Returns:
        "ipyleaflet.GeoJSON": ROIs as GeoJson layer styled with the provided styles
    """
    assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto map"
    if style is None:
        style = {
            "color": "black",
            "fill_color": "black",
            "fillOpacity": 0.1,
            "weight": 1,
        }
    if hover_style is None:
        hover_style = {}
    return GeoJSON(data=geojson, name=layer_name, style=style, hover_style=hover_style)


def apply_style_to_feature(feature: Dict, style: Dict) -> Dict:
    return {**feature, "properties": {**feature["properties"], "style": style}}


def filter_selected_features(features: list, selected_set: Set) -> list:
    return [
        feature for feature in features if feature["properties"]["id"] in selected_set
    ]


def convert_selected_set_to_geojson(
    features: list, selected_set: Set, style: Optional[Dict] = None
) -> Dict:
    """Returns a geojson dict containing a FeatureCollection for all the geojson objects in the
    selected_set
    Args:
        features (list): List of geojson features.
        selected_set (set): ids of selected geojson
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
    if not style:
        style = {
            "color": "blue",
            "weight": 2,
            "fillColor": "blue",
            "fillOpacity": 0.1,
        }
    selected_features = filter_selected_features(features, selected_set)
    styled_features = [apply_style_to_feature(f, style) for f in selected_features]
    return {"type": "FeatureCollection", "features": styled_features}


class Map_Controller:
    MAX_AREA = 100000000000  # UNITS = Sq. Meters
    MIN_AREA = 1000  # UNITS = Sq. Meters

    def __init__(self):
        self.factory = Factory()
        self.map = None
        self.draw_control = None
        self.selected_layer_name = "Selected"
        self.unselected_layer_name = "Unselected"
        # ids of items currently selected on map
        self.selected_set = set()
        # ids of shorelines currently selected on map
        self.selected_shorelines_set = set()
        # Basic settings and configurations
        self.settings = {
            "center_point": (36.8470, -121.8024),
            "zoom": 7,
            "draw_control": False,
            "measure_control": False,
            "fullscreen_control": False,
            "attribution_control": True,
            "Layout": Layout(width="100%", height="100px"),
        }
        self.callbacks = {"on_draw": []}
        #  create the map as well as the draw control
        self._init_map_components()

        # Warning and information boxes
        self._init_info_boxes()

    def register_callback(self, event, callback):
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def create_DrawControl(self, draw_control: DrawControl):
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

    def _init_map_components(self):
        """Initialize map-related attributes and settings."""
        self.map = self.create_map()
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.map.add(self.draw_control)
        self.map.add(LayersControl(position="topright"))

    def _init_info_boxes(self):
        """Initialize info and warning boxes for the map."""
        self.warning_box = HBox([])
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

    def remove_layer_by_name(self, layer_name: str):
        existing_layer = self.map.find_layer(layer_name)
        if existing_layer is not None:
            self.map.remove(existing_layer)

    def create_map(self):
        """create an interactive map object using the map_settings
        Returns:
           ipyleaflet.Map: ipyleaflet interactive Map object
        """
        return Map(
            draw_control=self.settings["draw_control"],
            measure_control=self.settings["measure_control"],
            fullscreen_control=self.settings["fullscreen_control"],
            attribution_control=self.settings["attribution_control"],
            center=self.settings["center_point"],
            zoom=self.settings["zoom"],
            layout=self.settings["Layout"],
            world_copy_jump=True,
        )

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

    def add_feature_on_map(
        self,
        new_feature,
        feature_name: str,
        layer_name: str = "",
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
        if hasattr(new_feature, "gdf"):
            bounds = new_feature.gdf.total_bounds
            self.map.zoom_to_bounds(bounds)
        self.load_on_map(new_feature, layer_name, on_hover, on_click)

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

    def create_layer_on_map(
        self,
        feature,
        layer_name: str,
        on_hover=None,
        on_click=None,
        style=None,
        hover_style=None,
    ) -> None:
        """
        Loads the provided feature on the map as a new layer, replacing any existing layer with the same name.

        Args:
            feature: The input feature which can be of type dict, gpd.GeoDataFrame or any object with an 'gdf' attribute.
            layer_name (str): The name of the layer.
            on_hover: Optional event handler for hover actions.
            on_click: Optional event handler for click actions.
            style (dict, optional): Styling attributes for the GeoJSON.
            hover_style (dict, optional): Styling for hovering attributes for the GeoJSON.

        Raises:
            Exception: If the feature is unsupported or empty.
        """

        # If the feature has an 'gdf' attribute, use it as the feature
        if hasattr(feature, "gdf"):
            feature = feature.gdf

        # Check if the feature is of valid type
        if isinstance(feature, (gpd.GeoDataFrame, dict)):
            new_layer = self.create_layer(
                feature, layer_name, style=style, hover_style=hover_style
            )

            if new_layer:  # Ensure new_layer is not None before replacing
                # Replace old feature layer with new feature layer
                self.replace_layer_by_name(
                    layer_name, new_layer, on_hover=on_hover, on_click=on_click
                )
        else:
            raise Exception("Unsupported feature type provided or feature is empty.")

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

        self.feature_html.value = (
            "<div style='max-width: 230px; max-height: 200px; overflow-x: auto; overflow-y: auto'>"
            "<b>Transect</b>"
            f"<p>Id: {transect_id}</p>"
            f"<p>Slope: {slope}</p>"
        )

    def create_bbox(
        self, geometry: dict | gpd.GeoDataFrame, layer_name: str = "Bbox", **kwargs
    ) -> Bounding_Box:
        # create bbox using factory
        new_feature = self.factory.make_feature(layer_name, geometry, **kwargs)
        if new_feature is None:
            return
        # this logic is now handled by the factory
        # bbox = Bounding_Box(geometry)
        # exception_handler.check_if_gdf_empty(bbox.gdf, "bounding box")

        # remove the existing bbox layer from map
        # self.remove_layer_by_name(layer_name)
        self.create_layer_on_map(
            new_feature,
            layer_name,
            style={
                "color": "#75b671",
                "fill_color": "#75b671",
                "opacity": 1,
                "fillOpacity": 0.1,
                "weight": 3,
            },
        )
        # Notify all registered callbacks for 'on_draw' event
        for callback in self.callbacks["on_draw"]:
            callback(new_feature)

    def handle_draw(self, draw_control: DrawControl, action: str, geo_json: dict):
        """Adds or removes the bounding box  when drawn/deleted from map
        Args:
            draw_control (ipyleaflet.leaflet.DrawControl): draw control used
            action (str): name of the most recent action ex. 'created', 'deleted'
            geo_json (dict): geojson dictionary
        """
        if (
            draw_control.last_action == "created"
            and draw_control.last_draw["geometry"]["type"] == "Polygon"
        ):
            # validate the bbox size
            geometry = draw_control.last_draw["geometry"]
            polygon_area = common.get_area(geometry)
            if polygon_area > Map_Controller.MAX_AREA:
                draw_control.clear()
                raise Exception(
                    f"Area too large. Must be smaller than {Map_Controller.MAX_AREA}"
                )
            elif polygon_area < Map_Controller.MIN_AREA:
                draw_control.clear()
                raise Exception(
                    f"Area too small. Must be smaller than {Map_Controller.MIN_AREA}"
                )
            else:
                # Remove old bbox from the map
                self.remove_layer_by_name("BBox")
                # add feature to map
                self.create_bbox(draw_control.last_draw["geometry"])
                # remove the user drawn bbox
                draw_control.clear()

    def create_layer(
        self, feature, layer_name: str, style: dict = None, hover_style: dict = None
    ) -> GeoJSON:
        """
        Creates a styled GeoJson layer based on the input feature.

        This function supports both dictionary (GeoJSON) and geodataframe input types for the feature.
        For dictionary input, it checks if the feature is not empty.
        For geodataframe input, it checks if the geodataframe is not empty and then converts it to GeoJSON format.
        If an unsupported feature type is provided or the feature is empty, the function returns None.

        Args:
            feature (dict or gpd.GeoDataFrame): The input feature, which can be a GeoJSON dictionary or a geodataframe.
            layer_name (str): The name of the GeoJSON layer.
            style (dict): Styling attributes for the GeoJSON.
            hover_style (dict): Hover styling attributes for the GeoJSON.

        Returns:
            ipyleaflet.GeoJSON or None: A styled GeoJson layer if successful, or None if the input is unsupported or empty.
        """

        # Convert feature's geodataframe attribute to GeoJSON if present
        if hasattr(feature, "gdf"):
            if feature.gdf.empty:
                print("Cannot add an empty geodataframe layer to the map.")
                return None
            layer_geojson = json.loads(feature.gdf.to_json())
            # convert layer to GeoJson and style it accordingly
            return feature.style_layer(layer_geojson, layer_name)

        # Handle dictionary or geodataframe directly passed
        if isinstance(feature, dict) and feature:
            layer_geojson = feature
        # Check if feature is a geodataframe
        elif isinstance(feature, gpd.GeoDataFrame):
            if feature.empty:
                print("Cannot add an empty geodataframe layer to the map.")
                return None
            layer_geojson = json.loads(feature.to_json())
        else:
            print("Unsupported feature type provided.")
            return None

        # Convert layer to GeoJson and style it accordingly
        styled_layer = style_layer(layer_geojson, layer_name, style, hover_style)
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
        self.remove_layer_by_name(self.selected_layer_name)

        layer = self.map.find_layer(self.unselected_layer_name)
        # get the features in the layer
        features = layer.data["features"]

        # create a new layer out of all the items in the layer that are in the selected set
        new_layer = GeoJSON(
            data=convert_selected_set_to_geojson(features, self.selected_set),
            name=self.selected_layer_name,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )

        self.replace_layer_by_name(
            self.selected_layer_name,
            new_layer,
            on_click=self.selected_onclick_handler,
        )

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
        self.remove_layer_by_name(self.selected_layer_name)

        # get the features in the unselected layer
        layer = self.map.find_layer(self.unselected_layer_name)
        features = layer.data["features"]

        # Recreate selected layers without layer that was removed
        selected_layer = GeoJSON(
            data=convert_selected_set_to_geojson(features, self.selected_set),
            name=self.selected_layer_name,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        self.replace_layer_by_name(
            self.selected_layer_name,
            selected_layer,
            on_click=self.selected_onclick_handler,
        )

    def remove_selected_shorelines(self) -> None:
        """Removes all the unselected rois from the map"""
        logger.info("Removing selected shorelines from map")
        # Remove the selected and unselected rois
        self.remove_layer_by_name(SELECTED_LAYER_NAME)
        self.remove_layer_by_name(Shoreline.LAYER_NAME)

        # Call the callback to update the selected shorelines
        #
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
