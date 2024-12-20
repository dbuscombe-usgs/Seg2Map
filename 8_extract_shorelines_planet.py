from coastseg import zoo_model
from coastseg.tide_correction import compute_tidal_corrections
from coastseg import file_utilities
from coastseg import classifier
import os
from coastseg import raster_utils
import numpy as np
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

# Assume the planet data is in this format
# sitename
# ├── jpg_files
# │   ├── preprocessed
# │   │   ├── RGB
# │   │   │   ├── 2020-06-01-21-16-34_RGB_PS.jpg
# │   │   │__NIR
# │   │   │   ├── 2020-06-01-21-16-34_NIR_PS.jpg
# │  
# │ _PS
# │   ├──meta
# │   │   ├── 20200601_211634_44_2277_3B_AnalyticMS_toar_clip.txt
# │   ├──ms
# │   │   ├── 20200601_211634_44_2277_3B_AnalyticMS_toar_clip.tif
# │   ├──cloud_mask
# │   │   ├── 20200601_211634_44_2277_3B_udm2_clip.tif

# This script assumes the site directory is in the format of the above structure and is provided
session_dir = r'C:\development\coastseg-planet\downloads\UNALAKLEET_pier_cloud_0.7_TOAR_enabled_2020-06-01_to_2023-08-01\e2821741-0677-435a-a93a-d4f060f3adf4\coregistered'
good_dir = r'C:\development\coastseg-planet\downloads\UNALAKLEET_pier_cloud_0.7_TOAR_enabled_2020-06-01_to_2023-08-01\e2821741-0677-435a-a93a-d4f060f3adf4\coregistered\segmentations\good'
# Make a simpler version of run model that doesn't need config

# Make a simpler version of extract shorelines that doesn't need config

zoo_model_instance = zoo_model.Zoo_Model()

# save settings to the zoo model instance
settings.update(model_setting)
# save the settings to the model instance
zoo_model_instance.set_settings(**settings)

model_implementation = settings.get('implementation', "BEST")
model_name = settings.get('model_type', None)
RGB_directory = os.path.join(session_dir, 'jpg_files', 'preprocessed', 'RGB')

output_directory = os.path.join(session_dir, 'segmentations')
os.makedirs(output_directory, exist_ok=True)


zoo_model_instance.run_model_on_directory(output_directory, RGB_directory,model_name )

# Now run the segmentation classifier on the output directory
classifier.filter_segmentations(output_directory,threshold=0.50)
