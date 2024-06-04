import logging
import os

from IPython.display import display
from coastseg import common, core_utilities, file_utilities, settings_UI, zoo_model
from coastseg.tide_correction import compute_tidal_corrections
from coastseg.upload_feature_widget import FileUploader
from ipyfilechooser import FileChooser
from ipywidgets import (
    Accordion,
    Button,
    Dropdown,
    FloatText,
    HBox,
    HTML,
    Layout,
    Output,
    Text,
    VBox,
)

# icons sourced from https://fontawesome.com/v4/icons/

logger = logging.getLogger(__name__)

# syle standards
MAX_HEIGHT='max-height: 300px;'
OVERFLOW_Y='overflow-y: auto;'
OVERFLOW_X='overflow-x: auto;'
TEXT_ALIGN='text-align: center;'

class UI_Models:
    # all instances of UI will share the same debug_view
    extract_shorelines_view = Output(layout={"border": "1px solid black"})
    tidal_correction_view = Output(layout={"border": "1px solid black"})

    def __init__(self):
        self.settings_dashboard = settings_UI.Settings_UI()
        self.zoo_model_instance = zoo_model.Zoo_Model()
        instructions = "Upload a GeoJSON file that contains either transects or shorelines.\
        If no file is provided, the CoastSeg will automatically attempt to load an available file for you.\
        In the event that no transects or shorelines are available within the specified region, an error will occur."
        self.fileuploader = FileUploader(
            title="",
            instructions=instructions,
            filter_pattern="*geojson",
            dropdown_options=["transects", "shorelines","shoreline extraction area"],
            file_selection_title="Select a geojson file",
            max_width=400,
        )
        # Controls size of ROIs generated on map
        self.model_dict = {
            "sample_direc": None,
            "use_GPU": "0",
            "implementation": "BEST",
            "model_type": "segformer_RGB_4class_8190958",
            "otsu": False,
            "tta": False,
            "img_type": "RGB",
        }
        # list of RGB and MNDWI models available
        self.RGB_models = [
            "segformer_RGB_4class_8190958",
            "sat_RGB_4class_6950472",
        ]
        # self.five_band_models = [
        #     "sat_5band_4class_7344606",
        # ]
        self.MNDWI_models = [
            "segformer_MNDWI_4class_8213443",
        ]
        self.NDWI_models = [
            "segformer_NDWI_4class_8213427",
        ]
        self.session_name = ""
        self.shoreline_session_directory = ""
        self.model_session_directory = ""
        self.shoreline_session_name = ""

        # Declare widgets and on click callbacks
        self._create_HTML_widgets()
        self._create_widgets()
        self._create_buttons()
        self.create_tidal_correction_widget()
        self.warning_row1 = HBox([])
        self.warning_row2 = HBox([])

    def get_warning_box(self, position: int = 1):
        if position == 1:
            return self.warning_row1
        if position == 2:
            return self.warning_row2

    def clear_extract_shorelines_btn(
        self,
    ):
        clear_button = Button(
            description="Clear TextBox", style=dict(button_color="#a3adac")
        )
        clear_button.on_click(self.clear_extract_shorelines_view)
        return clear_button


    def clear_tidal_correction_btn(
        self,
    ):
        clear_button = Button(
            description="Clear TextBox", style=dict(button_color="#a3adac")
        )
        clear_button.on_click(self.clear_tidal_correction_view)
        return clear_button


    def clear_extract_shorelines_view(self, btn):
        UI_Models.extract_shorelines_view.clear_output()

    def clear_tidal_correction_view(self, btn):
        UI_Models.tidal_correction_view.clear_output()

    def get_model_instance(self):
        return self.zoo_model_instance

    def set_session_name(self, name: str):
        self.session_name = str(name).strip()

    def get_session_name(
        self,
    ):
        return self.session_name

    def set_shoreline_session_name(self, name: str):
        self.shoreline_session_name = str(name).strip()

    def get_shoreline_session_name(
        self,
    ):
        return self.shoreline_session_name

    def get_shoreline_session_selection(self):
        output = Output()
        self.shoreline_session_name_text = Text(
            value="",
            placeholder="Enter name",
            description="Extracted Shoreline Session Name:",
            disabled=False,
            style={"description_width": "initial"},
        )

        enter_button = Button(description="Enter")

        @output.capture(clear_output=True)
        def enter_clicked(btn):
            session_name = str(self.shoreline_session_name_text.value).strip()
            base_path = os.path.abspath(core_utilities.get_base_dir())
            session_path = file_utilities.create_directory(base_path, "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(f"Session {session_name} already exists at {new_session_path} and will be overwritten.")
                self.set_shoreline_session_name(session_name)
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} will be created at {new_session_path}")
                self.set_shoreline_session_name(session_name)

        enter_button.on_click(enter_clicked)
        session_name_controls = HBox([self.shoreline_session_name_text, enter_button])
        return VBox([session_name_controls, output])

    def get_session_selection(self):
        output = Output()
        self.session_name_text = Text(
            value="",
            placeholder="Enter a session name",
            description="Session Name:",
            disabled=False,
            style={"description_width": "initial"},
        )

        enter_button = Button(description="Enter")

        @output.capture(clear_output=True)
        def enter_clicked(btn):
            session_name = str(self.session_name_text.value).strip()
            base_path = os.path.abspath(core_utilities.get_base_dir())
            session_path = file_utilities.create_directory(base_path, "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(f"Session {session_name} already exists at {new_session_path}")
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} will be created at {new_session_path}")
                self.set_session_name(session_name)

        enter_button.on_click(enter_clicked)
        session_name_controls = HBox([self.session_name_text, enter_button])
        return VBox([session_name_controls, output])

    def get_adv_model_settings_section(self):
        """
        Returns a VBox containing the advanced model settings widgets.
        The advanced model settings include the model implementation, Otsu threshold, and Test Time Augmentation.

        Returns:
            VBox: A VBox widget containing the advanced model settings widgets.
        """
        # declare settings widgets
        settings = {
            "model_implementation": self.model_implementation,
            "otsu": self.otsu_radio,
            "tta": self.tta_radio,
        }
        return VBox([widget for widget_name, widget in settings.items()])

    def get_basic_model_settings_section(self):
        """
        Returns a VBox containing the basic model settings widgets.

        This method creates and returns a VBox widget that contains the basic model settings widgets.
        The settings include the model input dropdown and the model dropdown.

        Returns:
            VBox: A VBox widget containing the basic model settings widgets.
        """
        # declare settings widgets
        settings = {
            "model_input": self.model_input_dropdown,
            "model_type": self.model_dropdown,
        }
        # create settings vbox
        return VBox([widget for widget_name, widget in settings.items()])

    def get_model_settings_accordion(self):
            """
            Returns an Accordion widget containing the basic and advanced model settings sections.

            Returns:
                Accordion: An Accordion widget with the basic and advanced model settings sections.
            """
            # create settings accordion widget
            settings_accordion = Accordion(
                children=[
                    self.get_basic_model_settings_section(),
                    self.get_adv_model_settings_section(),
                ]
            )
            settings_accordion.set_title(0, "Basic Model Settings")
            settings_accordion.set_title(1, "Advanced Model Settings")
            settings_accordion.selected_index = 0

            return settings_accordion

    def create_dashboard(self):
        layout = Layout(max_width='270px', overflow='auto',padding='0px 10px 0px 0px')
        # step 1: Select a Model
        step_1 = HBox([HBox([self.step_1_instr],layout = layout), self.get_model_settings_accordion()])
        # step 2: Select Settings
        step_2 = HBox(
                    [
                        VBox(
                            [
                                self.step_2_instr,
                                self.save_settings_btn,
                            ],
                            layout = layout
                        ),
                        self.settings_dashboard.render(),
                        self.get_view_settings_vbox(),
                    ]
                )
        
        # step 3 : Upload Files
        step_3 = HBox([HBox([self.step_3_instr],layout=layout), self.fileuploader.get_FileUploader_widget()])
        
        # step 4 : Extract Shorelines with Model
        step_4 = VBox([self.step_4_instr, self.get_session_selection(), self.use_select_images_button,self.extract_shorelines_button])
        
        # step 5 : Tidal Correction
        step_5 = VBox([self.step_5_instr, self.select_extracted_shorelines_session_button, self.create_tidal_correction_widget()])
        
        self.file_row = HBox([])
        self.extracted_shoreline_file_row = HBox([])
        self.tidal_correct_file_row = HBox([])
        display(
            step_1,
            step_2,
            step_3,
            step_4,
           HBox(
                [self.clear_extract_shorelines_btn(), UI_Models.extract_shorelines_view]
            ),
            self.file_row,
            self.warning_row1,
            self.line_widget,
            step_5,
            self.tidal_correct_file_row,
            HBox([self.clear_tidal_correction_btn(), UI_Models.tidal_correction_view]),
            self.warning_row2,
        )

    def create_tidal_correction_widget(self):
        load_style = dict(button_color="#69add1", description_width="initial")

        self.beach_slope_text = FloatText(value=0.02, description="Beach Slope (m/m):",style={'description_width': 'initial'})
        self.reference_elevation_text = FloatText(value=0.0, description="Beach Elevation (m, relative M.S.L.):",style={'description_width': 'initial'})

        self.tidally_correct_button = Button(
            description="Correct Tides",
            style=load_style,
            icon="fa-tint",
        )
        self.tidally_correct_button.on_click(self.tidally_correct_button_clicked)

        return VBox(
            [
                self.beach_slope_text,
                self.reference_elevation_text,
                self.tidally_correct_button,
            ]
        )

    @tidal_correction_view.capture(clear_output=True)
    def tidally_correct_button_clicked(self, button):
        # user must have selected imagery first
        # must select a directory of model outputs
        if self.shoreline_session_directory == "":
            self.launch_error_box(
                "Cannot correct tides",
                "Must click select session first",
                position=2,
            )
            return
        # get session directory location
        session_directory = self.shoreline_session_directory
        session_name = os.path.basename(session_directory)
        logger.info(f"session_directory: {session_directory}")
        # get roi_id
        # @todo find a better way to load the roi id
        config_json_location = file_utilities.find_file_recursively(
            session_directory, "config.json"
        )
        config = file_utilities.load_data_from_json(config_json_location)
        roi_id = config.get("roi_id", "")
        logger.info(f"roi_id: {roi_id}")

        beach_slope = self.beach_slope_text.value
        reference_elevation = self.reference_elevation_text.value
        # load in shoreline settings, session directory with model outputs, and a new session name to store extracted shorelines
        compute_tidal_corrections(
            session_name, [roi_id], beach_slope, reference_elevation
        )

    def _create_widgets(self):
        self.model_implementation = Dropdown(
            options=["ENSEMBLE", "BEST"],
            value="BEST",
            description="Select:",
            disabled=False,
        )
        self.model_implementation.observe(self.handle_model_implementation, "value")

        self.otsu_radio = Dropdown(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Otsu Threshold:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.otsu_radio.observe(self.handle_otsu, "value")

        self.tta_radio = Dropdown(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Test Time Augmentation:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.tta_radio.observe(self.handle_tta, "value")
        self.model_input_dropdown = Dropdown(
            options=["RGB", "MNDWI", "NDWI",],
            value="RGB",
            description="Model Input:",
            disabled=False,
        )
        self.model_input_dropdown.observe(self.handle_model_input_change, names="value")

        self.model_dropdown = Dropdown(
            options=self.RGB_models,
            value=self.RGB_models[0],
            description="Select Model:",
            disabled=False,
        )
        self.model_dropdown.observe(self.handle_model_type, "value")

    def _create_buttons(self):
        # button styles
        load_style = dict(button_color="#69add1", description_width="initial")
        action_style = dict(button_color="#ae3cf0")

        # Save settings button: saves both the model and extract shoreline settings
        self.save_settings_btn = Button(
            description="Save Settings",
            style=action_style,
            icon="floppy-o",
        )
        self.save_settings_btn.on_click(self.save_settings_clicked)
        
        # Extract Shorelines button: extracts shorelines from model outputs
        self.extract_shorelines_button = Button(
            description="Extract Shorelines", style=action_style,icon="fa-bolt",
        )
        self.extract_shorelines_button.on_click(self.run_model_button_clicked)

        # Select Images button: selects a directory of images to run the model on
        self.use_select_images_button = Button(
            description="Select Images",
            style=load_style,
            icon="fa-file-image-o",
        )
        self.use_select_images_button.on_click(self.use_select_images_button_clicked)

        self.select_extracted_shorelines_session_button = Button(
            description="Select Session",
            style=load_style,
        )
        self.select_extracted_shorelines_session_button.on_click(
            self.select_extracted_shorelines_button_clicked
        )

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.line_widget = HTML(
            value="____________________________________________________"
        )
        # step 1: Select a Model
        self.step_1_instr = HTML(
            value="<h2>Step 1: Select a Model</h2>\
            <b>1.</b> Select the Model Input: RGB, MNDWI or NDWI\
            <br> - The NDWI and MNDWI will be automatically created using available RGB imagery\
            <br><b>2.</b> Select a model.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )
        # step 2: Select Settings
        self.step_2_instr = HTML(
            value="<h2>Step 2: Select Settings</h2>\
            <b>1.</b> Adjust the settings used to extract shorelines from the satellite imagery\
            <br><b>2.</b> Click save to save the settings\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        ) 
        # step 3 : Upload Files
        self.step_3_instr = HTML(
            value="<h2>Step 3: Upload Files</h2>\
            - If both the transects and shorelines are in the same 'config_gdf.geojson' you will need to upload it for both reference shoreline and transects   \
            <br>1. <b>Upload a Reference Shoreline (Optional): </b> Upload GeoJSON file containing a reference shoreline\
            <br>2. <b>Upload a Transects (Optional): </b>Upload GeoJSON file containing transects\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )
        # step 4 : Extract Shorelines with Model
        self.step_4_instr = HTML(
            value="<h2>Step 4: Extract Shorelines with Model</h2>\
            <b>1. Session Name:</b> Enter a name.A folder with this name will be created in the 'sessions' directory.\
            <br><b>2. Select Images:</b> Select the RGB directory from a ROI directory with downloaded imagery from the 'data' directory.\
            <br><b>3. Run Model:</b> Click this button to apply the model on imagery. Outputs are saved under your session name in the 'sessions' directory.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )
        # step 5 : Tidal Correction
        self.step_5_instr = HTML(
            value="<h2>Step 5 : Tidal Correction</h2>\
            - The tide model must have been downloaded to CoastSeg/tide_model for tidal correction to work. Follow the guide to download the tide model: https://github.com/SatelliteShorelines/CoastSeg/wiki/09.-How-to-Download-Tide-Model \
            - Ensure shorelines are extracted prior to tidal correction. Not all imagery will contain extractable shorelines, thus, tidal correction may not be possible.\
            <br><b>1. Select a Session: </b> Choose a session from the 'sessions' directory containing extracted shorelines.\
            <br><b>2. Run Tidal Correction:</b> Runs the tide model and save tidally corrected CSV files in the selected session directory.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )     
        
        self.run_model_instr = HTML(
            value="<h2>Run a Model</h2>\
            <b>1. Session Name:</b> Enter a name.This creates a folder in the 'sessions' directory for model outputs.\
            <br><b>2. Select Images:</b> Choose an ROI directory with downloaded imagery from the 'data' directory.\
            <br><b>3. Run Model:</b> Click this button to apply the model on imagery. Extract Shorelines will be saved in the 'sessions' directory under the session name entered.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.extract_shorelines_instr = HTML(
            value="<h2>Extract Shorelines</h2>\
            Make sure to run the model before extracting shorelines.<br> \
            <b>1. Name Your Session:</b> Enter a name for your shoreline extraction session this will create a new folder in the 'sessions' directory.<br>\
            <b>2. Upload a File (Optional): </b> Upload a transects or a reference shoreline geojson file to extract shorelines with.<br>\
            <b>3. Choose Model Session</b> Select a directory from the 'sessions' directory which contains model outputs.<br>\
            <b>4. Click Extract Shorelines</b> Your extracted shorelines will be saved in the 'sessions' directory under your named session.<br><br>\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.tidally_correct_instr = HTML(
            value="<h2>Tidally Correct</h2>\
            Ensure shorelines are extracted prior to tidal correction. Not all imagery will contain extractable shorelines, thus, tidal correction may not be possible.\
            <br><b>1. Select a Session </b> Choose a session from the 'sessions' directory containing extracted shorelines.\
            <br><b>2. Correct Tides Button</b> Click to save tidally corrected CSV files in the selected session directory.\
            ",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

    def handle_model_implementation(self, change):
        self.model_dict["implementation"] = change["new"]

    def handle_model_type(self, change):
        # 2 class model has not been selected disable otsu threhold
        self.model_dict["model_type"] = change["new"]
        if "2class" not in change["new"]:
            if self.otsu_radio.value == "Enabled":
                self.model_dict["otsu"] = False
                self.otsu_radio.value = "Disabled"
            self.otsu_radio.disabled = True
        # 2 class model was selected enable otsu threhold radio button
        if "2class" in change["new"]:
            self.otsu_radio.disabled = False

    def handle_otsu(self, change):
        if change["new"] == "Enabled":
            self.model_dict["otsu"] = True
        if change["new"] == "Disabled":
            self.model_dict["otsu"] = False

    def handle_tta(self, change):
        if change["new"] == "Enabled":
            self.model_dict["tta"] = True
        if change["new"] == "Disabled":
            self.model_dict["tta"] = False

    def handle_model_input_change(self, change):
        if change["new"] == "RGB":
            self.model_dropdown.options = self.RGB_models
        if change["new"] == "MNDWI":
            self.model_dropdown.options = self.MNDWI_models
        if change["new"] == "NDWI":
            self.model_dropdown.options = self.NDWI_models
            
        self.model_dict["img_type"] = change["new"]
        # if change["new"] == "RGB+MNDWI+NDWI":
        #     self.model_dropdown.options = self.five_band_models

        
    @extract_shorelines_view.capture(clear_output=True)
    def run_model_button_clicked(self, button):
        # user must have selected imagery first
        if self.model_dict["sample_direc"] is None:
            self.launch_error_box(
                "Cannot Run Model",
                "You must click 'Select Images' first",
                position=1,
            )
            return
        # user must have selected imagery first
        session_name = self.get_session_name()
        if session_name == "":
            self.launch_error_box(
                "Cannot Run Model",
                "Must enter a session name first",
                position=1,
            )
            return
        if self.model_dict["sample_direc"] == "":
            self.launch_error_box(
                "Cannot Run Model",
                "Must click select images first and select a directory of jpgs.\n Example : C:/Users/username/CoastSeg/data/ID_lla12_datetime11-07-23__08_14_11/jpg_files/preprocessed/RGB/",
                position=1,
            )
            return
        
        print("Running the model. Please wait.")
        zoo_model_instance = self.get_model_instance()

        # get the transects and shorelines file paths that were uploaded
        transects_path = self.fileuploader.files_dict.get("transects", "")
        shoreline_path = self.fileuploader.files_dict.get("shorelines", "")
        shoreline_extraction_area_path = self.fileuploader.files_dict.get("shoreline extraction area", "")
        zoo_model_instance.run_model_and_extract_shorelines(
            self.model_dict["sample_direc"],
            session_name=session_name,
            shoreline_path=shoreline_path,
            transects_path=transects_path,
            shoreline_extraction_area_path = shoreline_extraction_area_path
        )


    @extract_shorelines_view.capture(clear_output=True)
    def select_RGB_callback(self, filechooser: FileChooser) -> None:
        """Handle the selection of a directory and check for the presence of JPG files.

        Args:
            filechooser: The file chooser widget used to select the directory.
        """
        from pathlib import Path

        if filechooser.selected:
            sample_direc = Path(filechooser.selected).resolve()
            print(f"The images in the folder will be segmented :\n{sample_direc} ")
            # Using itertools.chain to combine the results of two glob calls
            has_jpgs = any(sample_direc.glob("*.jpg")) or any(
                sample_direc.glob("*.jpeg")
            )
            if not has_jpgs:
                self.launch_error_box(
                    "No jpgs found",
                    f"The directory {sample_direc} contains no jpgs! Please select a directory with jpgs. Make sure to select 'RGB' folder located in the 'preprocessed' folder.'",
                    position=1,
                )
                self.model_dict["sample_direc"] = ""
            else:
                self.model_dict["sample_direc"] = str(sample_direc)
                self.zoo_model_instance.set_settings(sample_direc=str(sample_direc))
                self.save_updated_settings()

    @extract_shorelines_view.capture(clear_output=True)
    def use_select_images_button_clicked(self, button):
        # Prompt the user to select a directory of images
        file_chooser = common.create_dir_chooser(
            self.select_RGB_callback, title="Select directory of images"
        )
        # clear row and close all widgets in self.file_row before adding new file_chooser
        common.clear_row(self.file_row)
        # add instance of file_chooser to self.file_row
        self.file_row.children = [file_chooser]
        # udpate the settings in the model instance
        # self.save_updated_settings()
        # self.update_displayed_settings()


    @tidal_correction_view.capture(clear_output=True)
    def selected_shoreline_session_callback(self, filechooser: FileChooser) -> None:
        if filechooser.selected:
            self.shoreline_session_directory = os.path.abspath(filechooser.selected)

    @tidal_correction_view.capture(clear_output=True)
    def select_extracted_shorelines_button_clicked(self, button):
        # Prompt the user to select a directory of extracted shorelines session
        file_chooser = common.create_dir_chooser(
            self.selected_shoreline_session_callback,
            title="Select extracted shorelines session",
            starting_directory="sessions",
        )
        # clear row and close all widgets in self.tidal_correct_file_row before adding new file_chooser
        common.clear_row(self.tidal_correct_file_row)
        # add instance of file_chooser to self.tidal_correct_file_row
        self.tidal_correct_file_row.children = [file_chooser]

    def launch_error_box(self, title: str = None, msg: str = None, position: int = 1):
        # Show user error message
        warning_box = common.create_warning_box(
            title=title, msg=msg, instructions=None, msg_width="95%", box_width="30%"
        )
        # clear row and close all widgets before adding new warning_box
        common.clear_row(self.get_warning_box(position))
        # add instance of warning_box to warning_row
        self.get_warning_box(position).children = [warning_box]

    def save_updated_settings(self,):
        # get the settings from the settings dashboard
        settings = self.settings_dashboard.get_settings()
        # get the model settings
        settings.update(self.model_dict)
        # save the settings to the model instance
        self.zoo_model_instance.set_settings(**settings)
        # update the settings in the view settings section
        self.update_displayed_settings()

    @extract_shorelines_view.capture(clear_output=True)
    def save_settings_clicked(self, btn):
        try:
            # get the settings from the settings dashboard
            settings = self.settings_dashboard.get_settings()
            # get the model settings
            settings.update(self.model_dict)
            # save the settings to the model instance
            self.zoo_model_instance.set_settings(**settings)
            # update the settings in the view settings section
            self.update_displayed_settings()
        
        except Exception as error:
            self.launch_error_box(
                "Error saving settings",
                f"An error occurred while saving settings: {error}",
                position=1,
            )
            
    def get_view_settings_vbox(self) -> VBox:
        # update settings button
        action_style = dict(button_color="#ae3cf0")
        refresh_settings_btn = Button(
            description="Refresh Settings", icon="refresh", style=action_style
        )
        refresh_settings_btn.on_click(self.refresh_settings_btn_clicked)
        # get the settings from the model instance
        settings = self.zoo_model_instance.get_settings()
        setting_content = format_as_html(settings)
        self.settings_html = HTML(
             f"<div style='max-height: 300px;max-width: 280px; overflow-x: auto; overflow-y:  auto; text-align: left;'>"
            f"{setting_content}"
            f"</div>"
        )
        # The view settings section: contains the settings and a button to refresh the settings
        view_settings_vbox = VBox([self.settings_html, refresh_settings_btn])
        html_settings_accordion = Accordion(children=[view_settings_vbox],selected_index=0)
        html_settings_accordion.set_title(0, "View Settings")
        return html_settings_accordion

    @extract_shorelines_view.capture(clear_output=True)
    def refresh_settings_btn_clicked(self, btn):
        UI_Models.extract_shorelines_view.clear_output(wait=True)
        # refresh settings in view settings section
        try:
            self.update_displayed_settings()
        except Exception as error:
            self.launch_error_box(
                "Error saving settings",
                f"An error occurred while saving settings: {error}",
                position=1,
            )
            
    def update_displayed_settings(self):
        """
        Updates the displayed settings in the UI.

        Retrieves the settings from the `zoo_model_instance` and formats them as HTML.
        The formatted settings are then assigned to the `settings_html` widget.

        Returns:
            None
        """
        setting_content = format_as_html(self.zoo_model_instance.get_settings())
        self.settings_html.value = f"""<div style='max-height: 300px;max-width: 280px; overflow-x: auto; overflow-y:  auto; text-align: left;'>{setting_content}</div>"""

def format_as_html(settings: dict):
    """
    Generates HTML content displaying the settings.
    Args:
        settings (dict): The dictionary containing the settings.
    Returns:
        str: The HTML content representing the settings.
    """
    return f"""
    <h3>Model Settings</h3>
    <p>sample_direc: {settings.get("sample_direc", "unknown")}</p>
    <p>model_type: {settings.get("model_type", "unknown")}</p>
    <p>implementation: {settings.get("implementation", "unknown")}</p>
    <p>otsu enabled: {settings.get("otsu", "unknown")}</p>
    <p>tta enabled: {settings.get("tta", "unknown")}</p>
    <h3>Extract Shorelines Settings</h3>
    <p>cloud_thresh: {settings.get("cloud_thresh", "unknown")}</p>
    <p>dist_clouds: {settings.get("dist_clouds", "unknown")}</p>
    <p>output_epsg: {settings.get("output_epsg", "unknown")}</p>
    <p>save_figure: {settings.get("save_figure", "unknown")}</p>
    <p>min_beach_area: {settings.get("min_beach_area", "unknown")}</p>
    <p>min_length_sl: {settings.get("min_length_sl", "unknown")}</p>
    <p>cloud_mask_issue: {settings.get("cloud_mask_issue", "unknown")}</p>
    <p>sand_color: {settings.get("sand_color", "unknown")}</p>
    <p>max_dist_ref: {settings.get("max_dist_ref", "unknown")}</p>
    <p>Drop intersection points not on transects (drop_intersection_pts): {settings.get("drop_intersection_pts", "unknown")}</p>
    <h3>Advanced Settings</h3>
    <p>along_dist: {settings.get("along_dist", "unknown")}</p>
    <p>min_points: {settings.get("min_points", "unknown")}</p>
    <p>max_std: {settings.get("max_std", "unknown")}</p>
    <p>max_range: {settings.get("max_range", "unknown")}</p>
    <p>min_chainage: {settings.get("min_chainage", "unknown")}</p>
    <p>multiple_inter: {settings.get("multiple_inter", "unknown")}</p>
    <p>prc_multiple: {settings.get("prc_multiple", "unknown")}</p>
    """
    
