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

# Extract shorelines from directory
satellite = 'PS' # planet
meta_dir = os.path.join(session_dir, satellite, 'meta')
ms_dir = os.path.join(session_dir, satellite, 'ms')
udm2_dir = os.path.join(session_dir, satellite, 'udm2')
filepath =  [ms_dir,udm2_dir] # get the directories of the ms and udm2 files

pixel_size = 3.0 # meters For PlanetScope

# get the minimum beach area in number of pixels depending on the satellite
settings["min_length_sl"] 

# whatever file is currently being processed
filename =''

# get date
date = filename[:19]

# get filepath of ms and udm2 of this particular filenmae
fn = [os.path.join(ms_dir, filename), os.path.join(udm2_dir, filename)]


# basically this function should do what preprocess_single does

# filepaths to .tif files
fn_ms = fn[0]
fn_mask = fn[1]

im_ms, georef = raster_utils.read_ms_planet(fn_ms)
cloud_mask = raster_utils.read_cloud_mask_planet(fn_mask)
im_nodata = raster_utils.read_nodata_mask_planet(fn_mask)
combined_mask = np.logical_or(cloud_mask, im_nodata)

# Note may need to add code to add 0 pixels to the no data mask later

# Only do this is apply_cloud_mask is True
cloud_mask = np.logical_or(cloud_mask, im_nodata)

# Here is where we would get rid of the image if cloud cover or no data was above a certain threshold

# this is the next step in the process find the shorelines
    # ref_shoreline_buffer = SDS_shoreline.create_shoreline_buffer(
    #     cloud_mask.shape, georef, image_epsg, pixel_size, settings
    # )
    # # read the model outputs from the npz file for this image
    # npz_file = find_matching_npz(filename, os.path.join(session_path, "good"))
    # if npz_file is None:
    #     npz_file = find_matching_npz(filename, session_path)
    # if npz_file is None:
    #     logger.warning(f"npz file not found for {filename}")
    #     return None

    # # get the labels for water and land
    # land_mask = load_merged_image_labels(npz_file, class_indices=class_indices)
    # all_labels = load_image_labels(npz_file)

    # min_beach_area = settings["min_beach_area"]
    # land_mask = remove_small_objects_and_binarize(land_mask, min_beach_area)

    # # get the shoreline from the image
    # shoreline = find_shoreline(
    #     fn,
    #     image_epsg,
    #     settings,
    #     np.logical_xor(cloud_mask, im_nodata),
    #     cloud_mask,
    #     im_nodata,
    #     georef,
    #     land_mask,
    #     ref_shoreline_buffer,
    # )

# zoo_model_instance.run_model_and_extract_shorelines(
#             model_setting["sample_direc"],
#             session_name=model_session_name,
#             shoreline_path=shoreline_path,
#             transects_path=transects_path,
#             shoreline_extraction_area_path = shoreline_extraction_area_path,
#             coregistered=True,
#         )