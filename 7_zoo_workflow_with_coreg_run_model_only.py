import os
from coastseg import coastseg_logs
from coastseg import zoo_model
from coastseg.tide_correction import compute_tidal_corrections
from coastseg import file_utilities

# The Zoo Model is a machine learning model that can be used to extract shorelines from satellite imagery.
# This script will only run a single ROI at a time. If you want to run multiple ROIs, you will need to run this script multiple times.

# Extract Shoreline Settings
settings ={
    'min_length_sl': 100,       # minimum length (m) of shoreline perimeter to be valid
    'max_dist_ref':500,         # maximum distance (m) from reference shoreline to search for valid shorelines. This detrmines the width of the buffer around the reference shoreline  
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover (0-1). If the cloud cover is above this threshold, no shorelines will be extracted from that image
    'dist_clouds': 100,         # distance(m) around clouds where shoreline will not be mapped
    'min_beach_area': 50,      # minimum area (m^2) for an object to be labelled as a beach
    'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    "apply_cloud_mask": True,   # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
}


# The model can be run using the following settings:
model_setting = {
            "sample_direc": None, # directory of jpgs  ex. C:/Users/username/CoastSeg/data/ID_lla12_datetime11-07-23__08_14_11/jpg_files/preprocessed/RGB/",
            "use_GPU": "0",  # 0 or 1 0 means no GPU
            "implementation": "BEST",  # BEST or ENSEMBLE 
            "model_type": "global_segformer_RGB_4class_14036903", # model name from the zoo
            "otsu": False, # Otsu Thresholding
            "tta": False,  # Test Time Augmentation
        }
# Available models can run input "RGB" # or "MNDWI" or "NDWI"
img_type = "RGB"  # make sure the model name is compatible with the image type
# percentage of no data allowed in the image eg. 0.75 means 75% of the image can be no data
percent_no_data = 0.75

# 1. Set the User configuration Settings
# ---------------------------
# a. ENTER THE NAME OF THE SESSION TO SAVE THE MODEL PREDICTIONS TO
model_session_name = "coreg_session3"
# b. ENTER THE DIRECTORY WHERE THE INPUT IMAGES ARE STORED
# -  Example of the directory where the input images are stored ( this should be the /data folder in the CoastSeg directory)
# sample_directory = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52\jpg_files\preprocessed\RGB"
sample_directory = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original_mess_with\coregistered\jpg_files\preprocessed\RGB"

# 2. Save the settings to the model instance 
# -----------------
# Create an instance of the zoo model to run the model predictions
zoo_model_instance = zoo_model.Zoo_Model()
# Set the model settings to read the input images from the sample directory
model_setting["sample_direc"] = sample_directory
model_setting["img_type"] = img_type

# save settings to the zoo model instance
settings.update(model_setting)
# save the settings to the model instance
zoo_model_instance.set_settings(**settings)




# OPTIONAL: If you have a transects and shoreline file, you can extract shorelines from the zoo model outputs
transects_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52\transects.geojson" # path to the transects geojson file (optional, default will be loaded if not provided)
shoreline_path = "" # path to the shoreline geojson file (optional, default will be loaded if not provided)
shoreline_extraction_area_path= "" # path to the shoreline extraction area geojson file (optional)

# 3. Run the model and extract shorelines
# -------------------------------------



model_implementation = settings.get('implementation', "BEST")
use_GPU = settings.get('use_GPU', "0")
use_otsu = settings.get('otsu', False)
use_tta = settings.get('tta', False)


zoo_model_instance.run_model(
            img_type,
            model_implementation,
            session_name,
            input_directory,
            model_name=model_name,
            use_GPU=use_GPU,
            use_otsu=use_otsu,
            use_tta=use_tta,
            percent_no_data=percent_no_data,
            coregistered = True
)

# zoo_model_instance.run_model_and_extract_shorelines(
#             model_setting["sample_direc"],
#             session_name=model_session_name,
#             shoreline_path=shoreline_path,
#             transects_path=transects_path,
#             shoreline_extraction_area_path = shoreline_extraction_area_path
#         )

# 4. OPTIONAL: Run Tide Correction
# ------------------------------------------
# Tide Correction (optional)
# Before running this snippet, you must download the tide model to the CoastSeg/tide_model folder
# Tutorial: https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-and-clip-Tide-Model
#  You will need to uncomment the line below to run the tide correction

beach_slope = 0.02 # Slope of the beach (m)
reference_elevation = 0 # Elevation of the beach Mean Sea Level (M.S.L) (m)

# UNCOMMENT THESE LINES TO RUN THE TIDE CORRECTION
# roi_id = file_utilities.get_ROI_ID_from_session(session_name) # read ROI ID from the config.json file found in the extracted shoreline session directory
# compute_tidal_corrections(
#     session_name, [roi_id], beach_slope, reference_elevation
# )