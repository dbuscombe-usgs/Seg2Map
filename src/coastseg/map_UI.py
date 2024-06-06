# Standard Python imports
import os
import datetime
import logging

# Internal Python imports
from coastseg import exception_handler
from coastseg import common
from coastseg import core_utilities
from coastseg import file_utilities
from coastseg.extract_shorelines_widget import Extracted_Shoreline_widget


# External Python imports
import ipywidgets
import traitlets
from IPython.display import display
from ipyfilechooser import FileChooser
from google.auth import exceptions as google_auth_exceptions
from ipywidgets import Button
from ipywidgets import RadioButtons
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import HTML
from ipywidgets import BoundedFloatText
from ipywidgets import SelectMultiple
from ipywidgets import Output
from ipywidgets import FloatText
from ipywidgets import Accordion


from coastseg.settings_UI import Settings_UI

logger = logging.getLogger(__name__)

# icons sourced from https://fontawesome.com/v4/icons/

BOX_LAYOUT = Layout(
    width="350px",
    min_height="0px",  # Initial height
    max_height="250px",  # Maximum height
    flex_flow="row",
    overflow="auto",  # Will add scrollbar if content is too large
    display="flex",
    flex_grow=1,  # Allows the box to grow based on content
)

GRID_LAYOUT = Layout(
  display= 'grid',
  grid_template_rows='20px 30px',
  grid_auto_flow= 'column', 
  grid_auto_columns = '87px',
  width="550px",
)

def str_to_bool(var: str) -> bool:
    return var == "True"


def convert_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        logger.error(f"Invalid date: {date_str}. Expected format: 'YYYY-MM-DD'.{e}")
        raise ValueError(f"Invalid date: {date_str}. Expected format: 'YYYY-MM-DD'.{e}")


def format_as_html(settings: dict):
    """
    Generates HTML content displaying the settings.
    Args:
        settings (dict): The dictionary containing the settings.
    Returns:
        str: The HTML content representing the settings.
    """
    return f"""
    <h2>Settings</h2>
    <p>Satellites (sat_list): {settings.get("sat_list", "unknown")}</p>
    <p>dates: {settings.get("dates", "unknown")}</p>
    <p>Months to download (months_list): {settings.get("months_list", "unknown")}</p>
    <p>landsat_collection: {settings.get("landsat_collection", "unknown")}</p>
    <p>Maximum percentage of bad pixels (percent_no_data): {settings.get("percent_no_data", "unknown")}</p>
    <p>Maximum percentage of cloud pixels (cloud_thresh): {settings.get("cloud_thresh", "unknown")}</p>
    <p>Distance from clouds (dist_clouds): {settings.get("dist_clouds", "unknown")}</p>
    <p>output_epsg: {settings.get("output_epsg", "unknown")}</p>
    <p>save_figure: {settings.get("save_figure", "unknown")}</p>
    <p>Min beach area (min_beach_area): {settings.get("min_beach_area", "unknown")}</p>
    <p>Min Length of Shoreline (min_length_sl): {settings.get("min_length_sl", "unknown")}</p>
    <p>Apply cloud mask to images (apply_cloud_mask): {settings.get("apply_cloud_mask", "unknown")}</p>
    <p>image_size_filter: {settings.get("image_size_filter", "unknown")}</p>
    <p>Drop intersection points not on transects (drop_intersection_pts): {settings.get("drop_intersection_pts", "unknown")}</p>
    <p>cloud_mask_issue: {settings.get("cloud_mask_issue", "unknown")}</p>
    <p>sand_color: {settings.get("sand_color", "unknown")}</p>
    <p>Max distance from reference shoreline (max_dist_ref): {settings.get("max_dist_ref", "unknown")}</p>
    <p>Alongshore Distance (along_dist): {settings.get("along_dist", "unknown")}</p>
    <p>Minimum Number of Shoreline Points (min_points): {settings.get("min_points", "unknown")}</p>
    <p>Maximum STD of intersections (max_std): {settings.get("max_std", "unknown")}</p>
    <p>Max range of intersections (max_range): {settings.get("max_range", "unknown")}</p>
    <p>Minimum chainage (min_chainage): {settings.get("min_chainage", "unknown")}</p>
    <p>Multiple Intersections (multiple_inter): {settings.get("multiple_inter", "unknown")}</p>
    <p>Percentage Multiple (prc_multiple): {settings.get("prc_multiple", "unknown")}</p>
    """


class UI:
    # all instances of UI will share the same debug_view
    # this means that UI and coastseg_map must have a 1:1 relationship
    # Output widget used to print messages and exceptions created by CoastSeg_Map
    debug_view = Output(layout={"border": "1px solid black"})
    # Output widget used to print messages and exceptions created by download progress
    download_view = Output(layout={"border": "1px solid black"})
    preview_view = Output()

    def get_settings_dashboard(self, basic_settings: dict = {}):
        if not basic_settings:
            basic_settings = [
                "dates",
                "months_list",
                "percent_no_data",
                "max_dist_ref",
                "min_length_sl",
                "min_beach_area",
                "dist_clouds",
                "apply_cloud_mask",
                "cloud_thresh",
                "drop_intersection_pts",
            ]
        if not self.settings_dashboard:
            self.settings_dashboard = Settings_UI(basic_settings)
        return self.settings_dashboard

    def add_custom_widgets(self, settings_dashboard: Settings_UI):
        # create dropdown to select sand color
        sand_dropdown = ipywidgets.Dropdown(
            options=["default", "latest", "dark", "bright"],
            value="default",
            description="sand_color :",
            disabled=False,
        )
        # create dropdown to select mulitple satellites
        satellite_selection = ipywidgets.SelectMultiple(
            options=["L5", "L7", "L8", "L9", "S2"],
            value=["L8"],
            description="Satellites",
            disabled=False,
        )
        # create checkbox to control image size filter
        image_size_filter_checkbox = ipywidgets.Checkbox(
            value=True,
            description="Enable image size filter",
            indent=False,  # To align the description with the label
        )
        # create toggle to select cloud mask issue
        cloud_mask_issue = ipywidgets.ToggleButtons(
            options=["False", "True"],
            description=" Switch to True if sand pixels are masked (in black) on many images",
            disabled=False,
            button_style="",
            tooltips=[
                "No cloud mask issue",
                "Fix cloud masking",
            ],
        )
        settings_dashboard.add_custom_widget(
            satellite_selection,
            "sat_list",
            "Select Satellites",
            "Pick multiple satellites by holding the control key",
            advanced=False,
            index=2,
        )
         
        settings_dashboard.add_custom_widget(
            sand_dropdown,
            "sand_color",
            "Select Sand Color",
            "Sand color on beach for model to detect 'dark' (grey/black) 'bright' (white)",
            advanced=True,
            index=0,
        )
        settings_dashboard.add_custom_widget(
            cloud_mask_issue,
            "cloud_mask_issue",
            "Cloud Mask Issue",
            "Switch to True if sand pixels are masked (in black) on many images",
            advanced=True,
            index=-1,
        )

        settings_dashboard.add_custom_widget(
            image_size_filter_checkbox,
            "image_size_filter",
            "Image Size Filter",
            "Activate to filter out images that are smaller than 60% of the Region of Interest (ROI).",
            advanced=False,
            index=-1,
        )
        return settings_dashboard

    def __init__(self, coastseg_map, **kwargs):
        # save an instance of coastseg_map
        self.coastseg_map = coastseg_map
        # create the settings UI controller
        self.settings_dashboard = None
        self.settings_dashboard = self.get_settings_dashboard()
        # create custom widgets and add to settings dashboard
        self.settings_dashboard = self.add_custom_widgets(self.settings_dashboard)

        self.session_name = ""
        self.session_directory = ""

        # create an exception handler for extracted shorelines widget
        def my_exception_handler(error):
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        def clear_map_styles(error):
            self.coastseg_map.map.default_style = {"cursor": "default"}

        # create an exception handler for extracted shorelines widget
        def my_exception_handler(error):
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        def clear_map_styles(error):
            self.coastseg_map.map.default_style = {"cursor": "default"}

        # create the extract shorelines widget that controls shorelines on the map
        self.extract_shorelines_widget = Extracted_Shoreline_widget(
            coastseg_map.extract_shorelines_container
        )
        # register the exception handler with the extract shorelines widget
        self.extract_shorelines_widget.handle_exception.register_callback(
            my_exception_handler
        )
        self.extract_shorelines_widget.handle_exception.register_callback(
            clear_map_styles
        )

        # add callbacks to the extract shorelines widget
        # when the load button is clicked on the extract shorelines widget, load the selected shorelines on the map
        self.extract_shorelines_widget.add_load_callback(
            coastseg_map.load_selected_shorelines_on_map
        )
        # when the ROI is changed on the extract shorelines widget, update the map
        self.extract_shorelines_widget.add_ROI_callback(coastseg_map.update_extracted_shorelines_display)

        # when the delete button is clicked on the extract shorelines widget, remove the selected shorelines from the map
        self.extract_shorelines_widget.add_remove_all_callback(
            coastseg_map.delete_selected_shorelines
        )
        # when the extract_shorelines_widget needs to remove a layer from the map it will call this callback
        self.extract_shorelines_widget.add_remove_callback(
            coastseg_map.remove_layer_by_name
        )
        # link the widgets to the traitlets defined in the extract shorelines container located within coastseg_map
        coastseg_map.extract_shorelines_container.link_load_list(
            self.extract_shorelines_widget.load_list_widget
        )
        coastseg_map.extract_shorelines_container.link_trash_list(
            self.extract_shorelines_widget.trash_list_widget
        )
        coastseg_map.extract_shorelines_container.link_roi_list(
            self.extract_shorelines_widget.roi_list_widget
        )

        # create button styles
        self.create_styles()

        # buttons to load configuration files
        self.load_session_button = Button(
            description="Load Session", icon="files-o", style=self.load_style
        )
        self.load_session_button.on_click(self.on_load_session_clicked)

        self.settings_button = Button(
            description="Save Settings", icon="floppy-o", style=self.action_style
        )
        self.settings_button.on_click(self.save_settings_clicked)

        self.load_file_instr = HTML(
            value="<h2>Load Feature from File</h2>\
                 Load a feature onto map from geojson file.\
                ",
            layout=Layout(padding="0px"),
        )

        self.load_file_radio = ipywidgets.Dropdown(
            options=["Shoreline", "Transects", "Bbox", "ROIs"],
            value="Shoreline",
            description="",
            disabled=False,
        )
        self.load_file_button = Button(
            description=f"Load {self.load_file_radio.value} file",
            icon="file-o",
            style=self.load_style,
        )
        self.load_file_button.on_click(self.load_feature_from_file)

        def change_load_file_btn_name(change: dict):
            self.load_file_button.description = f"Load {str(change['new'])} file"

        self.load_file_radio.observe(change_load_file_btn_name, "value")

        # Generate buttons
        self.gen_button = Button(
            description="Generate ROI", icon="globe", style=self.action_style
        )
        self.gen_button.on_click(self.gen_roi_clicked)
        self.download_button = Button(
            description="Download Imagery", icon="download", style=self.action_style
        )
        self.download_button.on_click(self.download_button_clicked)

        self.preview_button = Button(
            description="Preview Imagery", icon="eye", style=self.action_style
        )
        self.preview_button.on_click(self.preview_button_clicked)

        self.extract_shorelines_button = Button(
            description="Extract Shorelines", style=self.action_style
        )
        self.extract_shorelines_button.on_click(self.extract_shorelines_button_clicked)

        # Clear  textbox button
        self.clear_debug_button = Button(
            description="Clear TextBox", style=self.clear_stlye
        )
        self.clear_debug_button.on_click(self.clear_debug_view)
        # Clear download messages button
        self.clear_downloads_button = Button(
            description="Clear Downloads", style=self.clear_stlye
        )
        self.clear_downloads_button.on_click(self.clear_download_view)

        # create the HTML widgets containing the instructions
        self._create_HTML_widgets()
        self.roi_slider_instr = HTML(value="<b>Choose Area of ROIs</b>")
        # controls the ROI units displayed
        self.units_radio = ipywidgets.Dropdown(
            options=["m²", "km²"],
            value="km²",
            description="Select Units:",
            disabled=False,
        )
        # create two float text boxes that will control size of ROI created
        self.sm_area_textbox = BoundedFloatText(
            value=0,
            min=0,
            max=98,
            step=1,
            description="Small ROI Area(km²):",
            style={"description_width": "initial"},
            disabled=False,
        )
        self.lg_area_textbox = BoundedFloatText(
            value=20,
            min=0,
            max=98,
            step=1,
            description="Large ROI Area(km²):",
            style={"description_width": "initial"},
            disabled=False,
        )

        # called when unit radio button is clicked
        def units_radio_changed(change: dict):
            """
            Change the maximum area allowed and the description of the small and large ROI area
            textboxes when the units radio is changed. When the units for area is m² the max ROI area size
            is 980000000 and when the units for area is m² max ROI area size
            is 98.

            Parameters:
            change (dict): event dictionary fired by clicking the units_radio button
            """
            try:
                MAX_AREA = 980000000
                STEP = 1000
                if change["name"] == "value":
                    if change["new"] == "m²":
                        MAX_AREA = 980000000
                        STEP = 1000
                        # convert km²to m²
                        new_sm_val = self.sm_area_textbox.value * (10**6)
                        new_lg_val = self.lg_area_textbox.value * (10**6)
                    elif change["new"] == "km²":
                        MAX_AREA = 98
                        STEP = 1
                        # convert m²to km²
                        new_sm_val = self.sm_area_textbox.value / (10**6)
                        new_lg_val = self.lg_area_textbox.value / (10**6)
                    # update the textboxes according to selected units
                    self.sm_area_textbox.max = MAX_AREA
                    self.sm_area_textbox.step = STEP
                    self.lg_area_textbox.max = MAX_AREA
                    self.lg_area_textbox.step = STEP
                    self.sm_area_textbox.value = new_sm_val
                    self.lg_area_textbox.value = new_lg_val
                    self.sm_area_textbox.description = (
                        f"Small ROI Area({self.units_radio.value}):"
                    )
                    self.lg_area_textbox.description = (
                        f"Large ROI Area({self.units_radio.value}):"
                    )
            except Exception as e:
                print(e)

        # when units radio button is clicked updated units for area textboxes
        self.units_radio.observe(units_radio_changed)

    def create_styles(self):
        """
        Initializes the styles used for various buttons in the user interface.
        Returns:
            None.
        """
        self.remove_style = dict(button_color="red")
        self.load_style = dict(button_color="#69add1", description_width="initial")
        self.action_style = dict(button_color="#ae3cf0")
        self.save_style = dict(button_color="#50bf8f")
        self.clear_stlye = dict(button_color="#a3adac")

    def launch_error_box(self, title: str = None, msg: str = None):
        # Show user error message
        warning_box = common.create_warning_box(title=title, msg=msg)
        # clear row and close all widgets in row before adding new warning_box
        common.clear_row(self.error_row)
        # add instance of warning_box to self.error_row
        self.error_row.children = [warning_box]

    def create_tidal_correction_widget(self, id_container: "traitlets.HasTraits"):
        load_style = dict(button_color="#69add1", description_width="initial")

        correct_tides_html = HTML(
            value="<h3><b>Apply Tide Correction</b></h3> \
               Only apply after extracting shorelines</br>",
            layout=Layout(margin="0px 5px 0px 0px"),
        )

        self.beach_slope_text = FloatText(value=0.02, description="Beach Slope (m/m):",style={'description_width': 'initial'})
        self.reference_elevation_text = FloatText(value=0.0, description="Beach Elevation (m, relative M.S.L.):",style={'description_width': 'initial'})

        self.scrollable_select = SelectMultiple(
            description="Select ROIs",
            options=id_container.ids,
            layout=Layout(overflow_y="auto", height="100px"),
        )

        # Function to update widget options when the traitlet changes
        def update_widget_options(change):
            self.scrollable_select.options = change["new"]

        # When the traitlet,id_container, trait 'ids' changes the update_widget_options will be updated
        id_container.observe(update_widget_options, names="ids")

        self.tidally_correct_button = Button(
            description="Correct Tides",
            style=load_style,
            icon="tint",
        )
        self.tidally_correct_button.on_click(self.tidally_correct_button_clicked)

        return VBox(
            [
                correct_tides_html,
                self.beach_slope_text,
                self.reference_elevation_text,
                self.scrollable_select,
                self.tidally_correct_button,
            ]
        )

    @debug_view.capture(clear_output=True)
    def tidally_correct_button_clicked(self, button):
        # get the selected ROI IDs if none selected give an error message
        selected_rois = self.scrollable_select.value
        if not selected_rois:
            self.launch_error_box(
                "Cannot correct tides",
                "Must enter a select an ROI ID first",
            )
            return

        print("Correcting tides... please wait")
        beach_slope = self.beach_slope_text.value
        reference_elevation = self.reference_elevation_text.value
        
        self.coastseg_map.compute_tidal_corrections(
            selected_rois, beach_slope, reference_elevation
        )
        # load in shoreline settings, session directory with model outputs, and a new session name to store extracted shorelines

    def set_session_name(self, name: str):
        self.session_name = str(name).strip()

    def get_session_name(
        self,
    ):
        return self.session_name

    def get_session_selection(self):
        output = Output()
        box_layout = Layout(
            min_height="0px",  # Initial height
            width="350px",
            max_height="50px",
            flex_flow="row",
            overflow="auto",
            display="flex",
            flex_grow=1,  # Allows the box to grow based on content
        )

        self.session_name_text = ipywidgets.Text(
            value="",
            placeholder="Enter a session name",
            description="Session Name:",
            disabled=False,
            style={"description_width": "initial"},
        )

        enter_button = ipywidgets.Button(
            description="Enter",
            layout=Layout(
                height="28px",
                width="80px",
            ),
        )

        @output.capture(clear_output=True)
        def enter_clicked(btn):
            # create the session directory
            session_name = str(self.session_name_text.value).strip()
            session_path = os.path.join( os.path.abspath(core_utilities.get_base_dir()), "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(f"Session {session_name} already exists. This session's data will be overwritten.")
                # print(f"Session {session_name} already exists. Name a new session.") # @dev might need this if we decide not to allow users to save to an existing session
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} was created.")
                new_session_path = file_utilities.create_directory(
                    session_path, session_name
                )
            self.coastseg_map.set_session_name(session_name)

        enter_button.on_click(enter_clicked)
        scrollable_output = ipywidgets.Box(children=[output], layout=box_layout)
        session_name_controls = HBox([self.session_name_text, enter_button])
        return VBox([session_name_controls, scrollable_output])

    def get_view_settings_vbox(self) -> VBox:
        # update settings button
        update_settings_btn = Button(
            description="Refresh Settings", icon="refresh", style=self.action_style
        )
        update_settings_btn.on_click(self.update_settings_btn_clicked)
        setting_content = format_as_html(self.coastseg_map.get_settings())
        self.settings_html = HTML(
             f"<div style='max-height: 300px;max-width: 280px; overflow-x: auto; overflow-y:  auto; text-align: left;'>"
            f"{setting_content}"
            f"</div>"
        )
        
        view_settings_vbox = VBox([self.settings_html, update_settings_btn])
        html_settings_accordion = Accordion(children=[view_settings_vbox])
        html_settings_accordion.set_title(0, "View Settings")
        return html_settings_accordion

    def save_to_file_buttons(self):
        # save to file buttons
        save_instr = HTML(
            value="<h2>Save to file</h2>\
                Save feature on the map to a geojson file.\
                <br>Geojson file will be saved to CoastSeg directory.\
            ",
            layout=Layout(padding="0px"),
        )

        self.save_radio = ipywidgets.Dropdown(
            options=[
                "Shoreline",
                "Transects",
                "Bbox",
                "ROIs",
            ],
            value="Shoreline",
            description="",
            disabled=False,
        )

        self.save_button = Button(
            description=f"Save {self.save_radio.value}",
            icon="floppy-o",
            style=self.save_style,
        )
        self.save_button.on_click(self.save_to_file_btn_clicked)

        def save_radio_changed(change: dict):
            self.save_button.description = f"Save {str(change['new'])} to file"

        self.save_radio.observe(save_radio_changed, "value")
        save_vbox = VBox([save_instr, self.save_radio, self.save_button])
        return save_vbox

    def load_feature_on_map_buttons(self):
        load_instr = HTML(
            value="<h2>Load Feature into Bounding Box</h2>\
                Loads shoreline or transects into bounding box on map.\
                </br>If no transects or shorelines exist in this area, then\
               </br> draw bounding box somewhere else\
                ",
            layout=Layout(padding="0px"),
        )
        self.load_radio = ipywidgets.Dropdown(
            options=["Shoreline", "Transects"],
            value="Transects",
            description="",
            disabled=False,
        )
        self.load_button = Button(
            description=f"Load {self.load_radio.value}",
            icon="file-o",
            style=self.load_style,
        )
        self.load_button.on_click(self.load_button_clicked)

        def handle_load_radio_change(change: dict):
            self.load_button.description = f"Load {str(change['new'])}"

        self.load_radio.observe(handle_load_radio_change, "value")
        load_buttons = VBox([load_instr, self.load_radio, self.load_button])
        return load_buttons

    def draw_control_section(self):
        load_instr = HTML(
            value="<h2>Draw Controls</h2>\
                Select to draw either a bounding box or shoreline extraction area on the map.\
                </br>Bounding boxes are green\
               </br>Shoreline extraction area is purple.(optional)\
                ",
            layout=Layout(padding="0px"),
        )
        # Draw controls
        self.draw_feature_controls = RadioButtons(
            options=["Bounding Box","Shoreline Extraction Area",],
            value="Bounding Box",
            description="Draw Controls:",
            disabled=False,
            layout={'width': 'max-content'},
            orientation="vertical",
        )

        self.draw_feature_controls.observe(self.on_draw_feature_controls_change, names="value")


        load_buttons = VBox([load_instr,self.draw_feature_controls,])
        return load_buttons

    def remove_buttons(self):
        # define remove feature radio box button
        remove_instr = HTML(
            value="<h2>Remove Feature from Map</h2>",
            layout=Layout(padding="0px"),
        )

        self.feature_dropdown = ipywidgets.Dropdown(
            options=[
                "Shoreline",
                "Transects",
                "Bbox",
                "ROIs",
                "Selected ROIs",
                "Selected Shorelines",
                "Extracted Shorelines",
                "Shoreline Extraction Area"
            ],
            value="Shoreline",
            description="",
            disabled=False,
        )
        self.remove_button = Button(
            description=f"Remove {self.feature_dropdown.value}",
            icon="ban",
            style=self.remove_style,
        )

        def handle_remove_radio_change(change: dict):
            self.remove_button.description = f"Remove {str(change['new'])}"

        self.remove_button.on_click(self.remove_feature_from_map)
        self.feature_dropdown.observe(handle_remove_radio_change, "value")
        # define remove all button
        self.remove_all_button = Button(
            description="Remove all", icon="trash-o", style=self.remove_style
        )
        self.remove_all_button.on_click(self.remove_all_from_map)

        remove_buttons = VBox(
            [
                remove_instr,
                self.feature_dropdown,
                self.remove_button,
                self.remove_all_button,
            ]
        )
        return remove_buttons

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.instr_create_roi = HTML(
            value="<h2><b>Generate ROIs on Map</b></h2> \
                 <li>  Draw a bounding box, then click Generate Roi\
                <li>ROIs (squares) are created within the bounding box along \n the shoreline.\
                <li>If no shoreline exists within the bounding box then ROIs cannot be created.\
                ",
            layout=Layout(margin="0px 5px 0px 0px"),
        )

        self.instr_download_roi = HTML(
            value="<h2><b>Download Imagery</b></h2> \
                <li><b>You must click an ROI on the map before you can download ROIs</b> \
                </br><h3><b><u>Where is my data?</u></b></br></h3> \
                <li>The data you downloaded will be in the 'data' folder in the main CoastSeg directory</li>\
                Each ROI you downloaded will have its own folder with the ROI's ID and\
                </br>the time it was downloaded in the folder name\
                </br><b>Example</b>: 'ID_1_datetime11-03-22__02_33_22'</li>\
                ",
            layout=Layout(margin="0px 0px 0px 5px"),
        )

        self.instr_config_btns = HTML(
            value="<h2><b>Load Sessions</b></h2>\
                <b>Load Session</b>: Load rois, shorelines, transects, and bounding box from session directory",
            layout=Layout(margin="0px 5px 0px 5px"),
        )  # top right bottom left

    def create_dashboard(self):
        """creates a dashboard containing all the buttons, instructions and widgets organized together."""
        # Buttons to load shoreline or transects in bbox on map
        load_buttons = self.load_feature_on_map_buttons()
        draw_control_section = self.draw_control_section()
        remove_buttons = self.remove_buttons()
        save_to_file_buttons = self.save_to_file_buttons()

        load_file_vbox = VBox(
            [self.load_file_instr, self.load_file_radio, self.load_file_button]
        )
        save_vbox = VBox(
            [
                save_to_file_buttons,
                load_file_vbox,
                remove_buttons,
                self.extract_shorelines_widget,
            ]
        )
        config_vbox = VBox(
            [
                self.instr_config_btns,
                self.load_session_button,
            ]
        )
        download_vbox = VBox(
            [
                self.instr_download_roi,
                self.download_button,
                self.preview_button,
                ipywidgets.Box(children=[UI.preview_view], layout=BOX_LAYOUT),
                self.extract_shorelines_button,
                self.get_session_selection(),
                self.create_tidal_correction_widget(self.coastseg_map.id_container),
                config_vbox,
            ]
        )

        area_control_box = VBox(
            [
                self.roi_slider_instr,
                self.units_radio,
                self.sm_area_textbox,
                self.lg_area_textbox,
            ]
        )
        ROI_btns_box = VBox([area_control_box, self.gen_button])
        roi_controls_box = VBox(
            [self.instr_create_roi, ROI_btns_box, load_buttons,draw_control_section],
            layout=Layout(margin="0px 5px 5px 0px"),
        )
        self.settings_row = HBox(
            [
                VBox(
                    [
                        self.settings_dashboard.render(),
                        self.settings_button,
                    ]
                ),
                self.get_view_settings_vbox(),
            ]
        )
        row_1 = HBox([roi_controls_box, save_vbox, download_vbox])
        # in this row prints are rendered with UI.debug_view
        row_2 = VBox([self.clear_debug_button, UI.debug_view])
        self.error_row = HBox([])
        self.file_chooser_row = HBox([])
        map_row = HBox([self.coastseg_map.map])
        download_msgs_row = HBox([self.clear_downloads_button, UI.download_view])

        return display(
            self.settings_row,
            row_1,
            row_2,
            self.error_row,
            self.file_chooser_row,
            map_row,
            download_msgs_row,
        )

    @debug_view.capture(clear_output=True)
    def update_settings_btn_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        # Update settings in view settings section
        try:
            setting_content = format_as_html(self.coastseg_map.get_settings())
            self.settings_html.value = f"""<div style='max-height: 300px;max-width: 280px; overflow-x: auto; overflow-y:  auto; text-align: left;'>{setting_content}</div>"""
        except Exception as error:
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def gen_roi_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        print("Generate ROIs button clicked")
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.gen_button.disabled = True
        # Generate ROIs along the coastline within the bounding box
        try:
            print("Generating ROIs please wait...")
            self.coastseg_map.load_feature_on_map(
                "rois",
                lg_area=self.lg_area_textbox.value,
                sm_area=self.sm_area_textbox.value,
                units=self.units_radio.value,
            )
        except Exception as error:
            print("ROIs could not be generated")
            self.coastseg_map.map.default_style = {"cursor": "default"}
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        else:
            self.coastseg_map.map.default_style = {"cursor": "default"}
            print("ROIs generated. Please Select at least one ROI on the map")
        self.coastseg_map.map.default_style = {"cursor": "default"}
        self.gen_button.disabled = False

    @debug_view.capture(clear_output=True)
    def load_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        try:
            if "shoreline" in btn.description.lower():
                print("Finding Shoreline")
                self.coastseg_map.load_feature_on_map("shoreline", zoom_to_bounds=True)
            if "transects" in btn.description.lower():
                print("Finding 'Transects'")
                self.coastseg_map.load_feature_on_map("transect", zoom_to_bounds=True)
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @debug_view.capture(clear_output=True)
    def save_settings_clicked(self, btn):
        # get the settings from the settings dashboard
        settings = self.settings_dashboard.get_settings()
        sat_list = settings.get("sat_list", [])
        if not sat_list:
            try:
                raise Exception("Must select at least one satellite first")
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        # save the settings to coastseg_map
        try:
            self.coastseg_map.set_settings(**settings)
            self.update_displayed_settings()
            # self.settings_html.value = format_as_html(self.coastseg_map.get_settings())
            
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)   

    def on_draw_feature_controls_change(self, change):
        if change["new"] == "Bounding Box":
            self.coastseg_map.drawing_shoreline_extraction_area = False
        else:
            self.coastseg_map.drawing_shoreline_extraction_area = True
            

    @debug_view.capture(clear_output=True)
    def extract_shorelines_button_clicked(self, btn):
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.extract_shorelines_button.disabled = True
        try:
            self.coastseg_map.extract_all_shorelines()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        self.extract_shorelines_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @preview_view.capture(clear_output=True)
    def preview_button_clicked(self, btn):
        UI.preview_view.clear_output()
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.preview_button.disabled = True
        UI.debug_view.append_stdout("Scroll down past map to see download progress.")
        try:
            try:
                self.preview_button.disabled = True
                self.coastseg_map.preview_available_images()
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        except google_auth_exceptions.RefreshError as exception:
            print(exception)
            exception_handler.handle_exception(
                exception,
                self.coastseg_map.warning_box,
                title="Authentication Error",
                msg="Please authenticate with Google using the cell above: \n Authenticate and Initialize with Google Earth Engine (GEE)",
            )
        self.preview_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @download_view.capture(clear_output=True)
    def download_button_clicked(self, btn):
        UI.download_view.clear_output()
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        self.download_button.disabled = True
        UI.debug_view.append_stdout("Scroll down past map to see download progress.")
        try:
            try:
                self.download_button.disabled = True
                self.coastseg_map.download_imagery()
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        except google_auth_exceptions.RefreshError as exception:
            print(exception)
            exception_handler.handle_exception(
                exception,
                self.coastseg_map.warning_box,
                title="Authentication Error",
                msg="Please authenticate with Google using the cell above: \n Authenticate and Initialize with Google Earth Engine (GEE)",
            )
        self.download_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    def clear_row(self, row: HBox):
        """close widgets in row/column and clear all children
        Args:
            row (HBox)(VBox): row or column
        """
        for index in range(len(row.children)):
            row.children[index].close()
        row.children = []

    @debug_view.capture(clear_output=True)
    def on_load_session_clicked(self, button):
        # Prompt user to select a config geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    self.coastseg_map.map.default_style = {"cursor": "wait"}
                    # load the session into coastseg_map and this should update the settings in coastseg_map
                    self.coastseg_map.load_fresh_session(filechooser.selected)
                    # update the session name text box with the session name
                    self.session_name_text.value = self.coastseg_map.get_session_name()
                    # update the settings dashboard with the settings from the loaded session
                    settings = self.coastseg_map.get_settings()
                    self.settings_dashboard.set_settings(settings)
                    self.update_displayed_settings()
                    self.coastseg_map.map.default_style = {"cursor": "default"}
            except Exception as error:
                # renders error message as a box on map
                self.coastseg_map.map.default_style = {"cursor": "default"}
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        # create instance of chooser that calls load_callback
        dir_chooser = common.create_dir_chooser(
            load_callback,
            title="Select Session Directory",
            starting_directory="sessions",
        )
        self.clear_row(self.file_chooser_row)
        # add instance of file_chooser to row 4
        self.file_chooser_row.children = [dir_chooser]
        self.coastseg_map.map.default_style = {"cursor": "default"}

    def update_displayed_settings(self):
        """
        Updates the displayed settings in the UI.

        Retrieves the settings from the `coastseg_map` and formats them as HTML.
        The formatted settings are then assigned to the `settings_html` widget.

        Returns:
            None
        """
        setting_content = format_as_html(self.coastseg_map.get_settings())
        self.settings_html.value = f"""<div style='max-height: 300px;max-width: 280px; overflow-x: auto; overflow-y:  auto; text-align: left;'>{setting_content}</div>"""


    @debug_view.capture(clear_output=True)
    def load_feature_from_file(self, btn):
        # Prompt user to select a geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    print(f"Loading {btn.description.lower()} this might take a few seconds...")
                    if "shoreline" in btn.description.lower():
                        logger.info(
                            f"Loading shoreline from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "shoreline",
                            os.path.abspath(filechooser.selected),
                            zoom_to_bounds=True,
                        )
                    if "transects" in btn.description.lower():
                        logger.info(
                            f"Loading transects from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "transect",
                            os.path.abspath(filechooser.selected),
                            zoom_to_bounds=True,
                        )
                    if "bbox" in btn.description.lower():
                        logger.info(
                            f"Loading bounding box from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "bbox",
                            os.path.abspath(filechooser.selected),
                            zoom_to_bounds=True,
                        )
                    if "rois" in btn.description.lower():
                        logger.info(
                            f"Loading ROIs from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "rois",
                            os.path.abspath(filechooser.selected),
                            zoom_to_bounds=True,
                        )
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        # change title of filechooser based on feature selected
        title = "Select a geojson file"
        # create instance of chooser that calls load_callback
        if "shoreline" in btn.description.lower():
            title = "Select shoreline geojson file"
        if "transects" in btn.description.lower():
            title = "Select transects geojson file"
        if "bbox" in btn.description.lower():
            title = "Select bounding box geojson file"
        if "rois" in btn.description.lower():
            title = "Select ROI geojson file"
        # create instance of chooser that calls load_callback
        file_chooser = common.create_file_chooser(load_callback, title=title)
        # clear row and close all widgets in row_4 before adding new file_chooser
        self.clear_row(self.file_chooser_row)
        # add instance of file_chooser to row 4
        self.file_chooser_row.children = [file_chooser]

    @debug_view.capture(clear_output=True)
    def remove_feature_from_map(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            # Prompt the user to select a directory of images
            if "extracted shorelines" in btn.description.lower():
                print(f"Removing extracted shoreline")
                self.coastseg_map.remove_extracted_shoreline_layers()
            elif "selected shorelines" in btn.description.lower():
                print(f"Removing Selected Shorelines")
                self.coastseg_map.remove_selected_shorelines()
            elif "shoreline extraction area" in btn.description.lower():
                print(f"Removing Shoreline Extraction Area")
                self.coastseg_map.remove_shoreline_extraction_area()
            elif "selected rois" in btn.description.lower():
                print(f"Removing Selected ROIs")
                self.coastseg_map.remove_selected_rois()
            elif "shoreline" in btn.description.lower():
                print(f"Removing shoreline")
                self.coastseg_map.remove_shoreline()
            elif "transects" in btn.description.lower():
                print(f"Removing  transects")
                self.coastseg_map.remove_transects()
            elif "bbox" in btn.description.lower():
                print(f"Removing bounding box")
                self.coastseg_map.remove_bbox()
            elif "rois" in btn.description.lower():
                print(f"Removing ROIs")
                self.coastseg_map.remove_all_rois()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def save_to_file_btn_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            if "shoreline" in btn.description.lower():
                print(f"Saving shoreline to file")
                self.coastseg_map.save_feature_to_file(
                    self.coastseg_map.shoreline, "shoreline"
                )
            if "transects" in btn.description.lower():
                print(f"Saving transects to file")
                self.coastseg_map.save_feature_to_file(
                    self.coastseg_map.transects, "transects"
                )
            if "bbox" in btn.description.lower():
                print(f"Saving bounding box to file")
                self.coastseg_map.save_feature_to_file(
                    self.coastseg_map.bbox, "bounding box"
                )
            if "rois" in btn.description.lower():
                print(f"Saving ROIs to file")
                self.coastseg_map.save_feature_to_file(self.coastseg_map.rois, "ROI")
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def remove_all_from_map(self, btn):
        try:
            self.coastseg_map.remove_all()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()

    def clear_download_view(self, btn):
        UI.download_view.clear_output()
