import os
from coastseg import coastseg_logs
from coastseg.common import initialize_gee
from coastseg import coastseg_map
from coastseg import core_utilities

base_dir = core_utilities.get_base_dir()
print(f"The base directory is {base_dir}")
# path to Coastseg/data
data_folder = base_dir / "data"
print(f"The data directory is {data_folder}")

# if you get an error here, enter your project id
initialize_gee(auth_mode = "localhost",project='')
# initialize the CoastSeg map
coastsegmap=coastseg_map.CoastSeg_Map(create_map=False)

# Construct the path to rois.geojson
rois_path = os.path.join(os.path.abspath(base_dir), 'examples',"rois.geojson")
print(f"Loading ROIs from {rois_path}")

# sample ROI (Region of Interest) file
roi = coastsegmap.load_feature_from_file('roi',rois_path)
print(roi)
# get the select all the ROI IDs from the file and store them in a list
roi_ids =  list(roi.gdf.id)
print(f"Downloading imagery for ROI with ID {roi_ids}")
# customize the settings for the imagery download
settings = {
    'sat_list':['S2'],                    # list of satellites to download imagery from. Options: 'L5', 'L7', 'L8', 'L9','S2'
    'dates':["2024-02-01", "2024-04-01"], # Start and end date to download imagery
    'landsat_collection':'C02',           # GEE collection to use. CoastSeg uses the Landsat Collection 2 (C02) by default
    "image_size_filter": True,            # filter images into bad folder if the images are less than 60% of the expected area. If False, no images will be filtered
    "apply_cloud_mask": True,             # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
    }

# # download the imagery for that ROI to the /data folder
coastsegmap.download_imagery(rois=roi.gdf,selected_ids=roi_ids,settings=settings,file_path=data_folder)

# name the session where you want to save the extracted shorelines
# session_name = 'sample_session1'
session_name = 'sample_session1'
coastsegmap.set_session_name(session_name)

# Modify the settings for the shoreline extraction here
# These settings will only extract shorelines with:
# - a minimum beach area of 500 m^2
# - a minimum length of 20 m
# - a maximum distance from the reference shoreline of 300 m
coastsegmap.set_settings(min_beach_area=100,min_length_sl=20,max_dist_ref=300)

# load a shoreline file from the examples folder
shoreline_path = os.path.join(base_dir, 'examples', "shoreline.geojson")
shoreline = coastsegmap.load_feature_from_file('shoreline',shoreline_path)

# load transects from the examples folder
transect_path = os.path.join(base_dir,'examples', "transects.geojson")
transects = coastsegmap.load_feature_from_file('transects', transect_path)

# extract the shorelines for the selected ROI and save them to the /sessions/session_name folder
coastsegmap.extract_all_shorelines(roi_ids = roi_ids)

# Tide Correction (optional)
# Tutorial: https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-and-clip-Tide-Model
# Tide Model must be downloaded to CoastSeg/tide_model
beach_slope = 0.02 # Slope of the beach (m)
reference_elevation = 0 # Elevation of the beach Mean Sea Level (M.S.L) (m)
# coastsegmap.compute_tidal_corrections(roi_ids, beach_slope, reference_elevation)
