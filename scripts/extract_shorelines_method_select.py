import os
from coastseg import coastseg_logs
from coastseg import coastseg_map
from coastseg import core_utilities

# How to run this script
# 1. Enter the name of the session you want to extract shorelines from for base_session_name

def load_base_session(session_name):
    # session_name = 'paper_dfg2_extract_shorelines_buffer_62'
    session_path = os.path.join(os.path.abspath(base_dir),'sessions', session_name)
    print(f"Loading session from {session_path}")
    coastsegmap.load_fresh_session(session_path)

def extract_shorelines_for_new_session(session_name,new_settings,apply_tide_correction=False,beach_slope=0.02, reference_elevation=0):
    """
    Extracts shorelines for a new session and optionally applies tide correction.
    Parameters:
        session_name (str): The name of the new session where the extracted shorelines will be saved. If not renamed, it will overwrite the existing session.
        new_settings (dict): A dictionary containing the new settings to be applied.
        apply_tide_correction (bool, optional): If True, applies tide correction. Default is False.
        beach_slope (float, optional): The slope of the beach used for tide correction. Default is 0.02.
        reference_elevation (float, optional): The reference elevation used for tide correction. Default is 0.
    Returns:
    None
    """
    # name the new session where you want to save the new extracted shorelines to. If you don't rename the session, it will overwrite the existing session
    coastsegmap.set_session_name(session_name)
    coastsegmap.set_settings(**new_settings)


    # get the ROI from the loaded session
    roi= coastsegmap.rois
    # get the select all the ROI IDs from the file and store them in a list
    roi_ids =  list(roi.gdf.id)

    print(f"Extracting shorelines for ROI with ID {roi_ids}")

    # extract the shorelines for the selected ROI and save them to the /sessions/session_name folder
    coastsegmap.extract_all_shorelines(roi_ids = roi_ids)

    if apply_tide_correction:
        # Tide Correction (optional)
        # Before running this snippet, you must download the tide model to the CoastSeg/tide_model folder
        # Tutorial: https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-and-clip-Tide-Model
        coastsegmap.compute_tidal_corrections(roi_ids, beach_slope, reference_elevation)


base_dir = core_utilities.get_base_dir()
# create a new coastseg_map object
coastsegmap=coastseg_map.CoastSeg_Map(create_map=False)


# 1. Enter the name of the session you want to load here
base_session_name = 'sample_session1'
# 2. load the base session. This clears out the previous session and loads the new session
load_base_session(base_session_name)

#3. Name one version of the session for the one's that extract shorelines with "find_wl_contours1" method AKA"mndwi method"
session_name = base_session_name + "_find_wl_contours1"

#4. Here are the new settings we want to apply to the new session. We are changing the contours_method to 'find_wl_contours1'
new_settings = { 
    'contours_method': 'find_wl_contours1', # 'find_wl_contours1' or 'find_wl_contours2'
}
#5. Extract shorelines for the new session using the new settings
extract_shorelines_for_new_session(session_name,new_settings,apply_tide_correction=False)


# Now lets do the same for the other method
# 1. Load the base session
load_base_session(base_session_name)

# 2. Name the new session for the one's that extract shorelines with "find_wl_contours2" method AKA "ML thresholding method"
# Name one version of the session for the one's that extract shorelines with "find_wl_contours1" method AKA"mndwi method"
session_name = base_session_name + "_find_wl_contours2"

#3. Here are the new settings we want to apply to the new session. We are changing the contours_method to 'find_wl_contours2'
new_settings = { 
    'contours_method': 'find_wl_contours2', # 'find_wl_contours1' or 'find_wl_contours2'
}
#4. Extract shorelines for the new session using the new settings
extract_shorelines_for_new_session(session_name,new_settings,apply_tide_correction=False)
