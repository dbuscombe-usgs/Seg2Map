# Standard library imports
import colorsys
import traceback
import copy
import shapely
import shutil
import fnmatch
import json
import json
import logging
import re
import os
import datetime
from glob import glob
from time import perf_counter
from typing import Any, Optional, Union, List, Dict
from time import perf_counter
from typing import Dict, List, Optional, Union
from itertools import islice

# External dependencies imports
import dask
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
import numpy as np
from ipyleaflet import GeoJSON
from skimage import measure, morphology
import skimage.measure as measure
import skimage.morphology as morphology
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.gridspec as gridspec
from shapely.geometry import MultiPoint, LineString

# coastsat imports
from coastsat import SDS_preprocess, SDS_shoreline, SDS_tools
from coastseg import geodata_processing
from coastseg import file_utilities
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.measure as measure
import skimage.morphology as morphology

from coastsat.SDS_download import get_metadata
from coastsat import SDS_download
from coastsat.SDS_tools import (
    get_filenames,
    get_filepath,
    output_to_gdf,
    remove_duplicates,
    remove_inaccurate_georef,
)
# from coastsat.SDS_transects import compute_intersection_QC
from ipyleaflet import GeoJSON
from matplotlib import gridspec
from matplotlib.colors import rgb2hex
from matplotlib.pyplot import get_cmap
from skimage import measure, morphology
from tqdm.auto import tqdm

# Internal dependencies imports
from coastseg import common, exceptions
from coastseg.validation import get_satellites_in_directory
from coastseg.filters import filter_model_outputs, apply_land_mask
from coastseg.common import get_filtered_files_dict, edit_metadata

from scipy.spatial import KDTree
from shapely.geometry import LineString
import pytz

# Set pandas option
pd.set_option("mode.chained_assignment", None)

# Logger setup
logger = logging.getLogger(__name__)
__all__ = ["Extracted_Shoreline"]


def time_func(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds to run.")
        # logger.debug(f"{func.__name__} took {end - start:.6f} seconds to run.")
        return result

    return wrapper


def save_shorelines_to_geojson(all_shorelines_gdf: gpd.GeoDataFrame, session_path: str) -> Dict[str, str]:
    """
    Processes and saves shoreline data to GeoJSON files.

    Args:
        all_shorelines_gdf (geopandas.GeoDataFrame): GeoDataFrame containing shoreline data.
        session_path (str): The directory path where the GeoJSON files will be saved.

    Returns:
        Dict[str, str]: A dictionary with paths to the saved GeoJSON files.
    """
    all_shorelines_gdf = all_shorelines_gdf.reset_index(drop=True)
    
    # Convert to EPSG 4326
    all_shorelines_gdf_4326 = all_shorelines_gdf.to_crs(epsg=4326)
    # Drop the filename column
    all_shorelines_gdf_4326.drop(columns=["filename"], inplace=True)

    if all_shorelines_gdf_4326.empty:
        print("No shorelines were extracted.")
        logger.warning("No shorelines were extracted.")
        return {}

    # Save extracted shorelines to GeoJSON files
    lines_geojson_path = os.path.join(session_path, 'extracted_shorelines_lines.geojson')
    all_shorelines_gdf_4326.to_file(lines_geojson_path, driver="GeoJSON")
    print(f"extracted_shorelines_lines.geojson saved to {lines_geojson_path}")

    # Convert linestrings to multipoints
    points_gdf = convert_linestrings_to_multipoints(all_shorelines_gdf_4326)
    projected_gdf = stringify_datetime_columns(points_gdf)
    
    # Save extracted shorelines as a GeoJSON file
    points_geojson_path = os.path.join(session_path, 'extracted_shorelines_points.geojson')
    projected_gdf.to_file(points_geojson_path, driver="GeoJSON")
    print(f"extracted_shorelines_points.geojson saved to {points_geojson_path}")

    return {
        "lines_geojson": lines_geojson_path,
        "points_geojson": points_geojson_path
    }

def aggregate_shoreline_data(shoreline_geodataframe: gpd.GeoDataFrame) -> Dict[str, List[Any]]:
    """
    Aggregates shoreline data from a GeoDataFrame, extracting contours and appending relevant metadata.

    Args:
        shoreline_geodataframe (geopandas.GeoDataFrame): GeoDataFrame containing shoreline data.
        contour_extractor (callable): Function to extract contours from shorelines.

    Returns:
        Dict[str, List[Any]]: Dictionary containing processed shorelines and metadata.
    """
    shoreline_dict = {
        "dates": [],
        "shorelines": [],
        "filename": [],
        "satname": [],
        "cloud_cover": [],
        "geoaccuracy": [],
        "idx": [],
        "MNDWI_threshold": [],
    }
    # convert the geodataframes to a dict with one list sorted by date with all the shorelines
    for date, group in shoreline_geodataframe.groupby("date"):
        shorelines = [np.array(geom.coords) for geom in group.geometry]
        contours_array = extract_contours(shorelines)
        shoreline_dict["shorelines"].append(contours_array)
        shoreline_dict["dates"].append(date)
        # Append values for each group, ensuring they are correctly extracted
        shoreline_dict["cloud_cover"].append(group["cloud_cover"].values[0])
        shoreline_dict["geoaccuracy"].append(group["geoaccuracy"].values[0])
        shoreline_dict["idx"].append(group["idx"].values[0])
        shoreline_dict["filename"].append(group["filename"].values[0])
        shoreline_dict["satname"].append(group["satname"].values[0])
        shoreline_dict["MNDWI_threshold"].append(group["MNDWI_threshold"].values[0])

    return shoreline_dict

def process_shoreline(
    contours: List[np.ndarray], 
    cloud_mask: np.ndarray, 
    im_nodata: np.ndarray, 
    georef: Dict[str, Any], 
    image_epsg: int, 
    settings: Dict[str, Any], 
    date: Union[str, datetime.datetime], 
    **kwargs: Any
) -> gpd.GeoDataFrame:
    """
    Process the shoreline data by converting coordinates, filtering points, and creating a GeoDataFrame.

    Args:
        contours (list): List of contours representing the shoreline in pixel coordinates.
        cloud_mask (numpy.ndarray): Binary mask indicating cloud pixels.
        im_nodata (numpy.ndarray): Binary mask indicating no data pixels.
        georef (dict): Georeferencing information.
        image_epsg (int): EPSG code of the image.
        settings (dict): Dictionary of processing settings.
        date (str or datetime.datetime): Date of the shoreline data.
        **kwargs: Additional keyword arguments.

    Returns:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing the processed shoreline data.

    """
    # convert the contours that are currently pixel coordinates to world coordiantes
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    contours_epsg = SDS_tools.convert_epsg(
        contours_world, image_epsg, settings["output_epsg"]
    )
    # this is the shoreline in the form of a list of numpy arrays, each array containing the coordinates of a shoreline x,y,z
    contours_long = filter_contours_by_length(contours_epsg, settings["min_length_sl"])
    # this removes the z coordinate from each shoreline point, so the format is list of numpy arrays, each array containing the x,y coordinates of a shoreline point
    contours_2d = [contour[:, :2] for contour in contours_long]
    # remove shoreline points that are too close to the no data mask
    new_contours = filter_points_within_distance_to_mask(
        contours_2d,
        im_nodata,
        georef,
        image_epsg,
        settings["output_epsg"],
        distance_threshold=60,
    )
    # remove shoreline points that are too close to the cloud mask
    new_contours = filter_points_within_distance_to_mask(
        new_contours,
        cloud_mask,
        georef,
        image_epsg,
        settings["output_epsg"],
        distance_threshold=settings["dist_clouds"],
    )
    filtered_contours_long = filter_contours_by_length(
        new_contours, settings["min_length_sl"]
    )
    contours_shapely = [LineString(contour) for contour in filtered_contours_long]
    if isinstance(date, str):
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d-%H-%M-%S")
    else:
        date_obj = date

    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
    # compute updated cloud cover percentage (without no data pixels)
    valid_pixels = np.sum(~im_nodata)
    cloud_cover = np.sum(cloud_mask_adv.astype(int)) / valid_pixels.astype(int)

    gdf = gpd.GeoDataFrame(
        {
            "date": np.tile(date_obj, len(contours_shapely)),
            "cloud_cover": np.tile(cloud_cover, len(contours_shapely)),
        },
        geometry=contours_shapely,
        crs=f"EPSG:{image_epsg}",
    )
    return gdf



def extract_shorelines_coastsat(
    metadata:dict,
    settings:dict,
    output_directory: str = None,
    shoreline_extraction_area: gpd.GeoDataFrame = None,
    geoaccuracy_threshold:float=10,
):
    """
    Extract shorelines from satellite images

    A modified version of the original extract shorelines by KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the image
        'min_beach_area': int
            minimum allowable object area (in metres^2) for the class 'sand',
            the area is converted to number of connected pixels
        'min_length_sl': int
            minimum length (in metres) of shoreline contour to be valid
        'sand_color': str
            default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline
        'adjust_detection': bool
            if True, allows user to manually adjust the detected shoreline
        'pan_off': bool
            if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
        's2cloudless_prob': float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask
    output_directory: str (default: None)
        The directory to save the output files. If None, the output files will be saved in the same directory as the input files.
    shoreline_extraction_area: gpd.GeoDataFrame (default: None)
        A geodataframe containing polygons indicating the areas to extract the shoreline from. Any shoreline outside of these polygons will be discarded.

    Returns:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates + metadata

    """

    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    collection = settings["inputs"]["landsat_collection"]

    sitename_location = os.path.join(filepath_data, sitename)
    # set up logger at the output directory if it is provided otherwise set up logger at the sitename location
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        logger = SDS_download.setup_logger(
            output_directory,
            "extract_shorelines_report",
            log_format="%(levelname)s - %(message)s",
        )
    else:
        logger = SDS_download.setup_logger(
            sitename_location,
            "extract_shorelines_report",
            log_format="%(levelname)s - %(message)s",
        )

    logger.info(f"Please read the following information carefully:\n")
    logger.info(
        "find_wl_contours2: A method for extracting shorelines that uses the sand water interface detected with the model to refine the threshold that's used to detect shorelines .\n  - This is the default method used when there are enough sand pixels within the reference shoreline buffer.\n"
    )
    logger.info(
        "find_wl_contours1: This shoreline extraction method uses a threshold to differentiate between water and land pixels in images, relying on Modified Normalized Difference Water Index (MNDWI) values. However, it may inaccurately classify snow and ice as water, posing a limitation in certain environments.\n  - This is only used when not enough sand pixels are detected within the reference shoreline buffer.\n"
    )
    logger.info(
        "---------------------------------------------------------------------------------------------------------------------"
    )
    all_shorelines = []
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)
    # close all open figures
    plt.close("all")
    output_epsg = settings["output_epsg"]
    default_min_length_sl = settings["min_length_sl"]
    # loop through satellite list
    for satname in metadata.keys():
        # get images
        filepath = SDS_tools.get_filepath(settings["inputs"], satname)
        filenames = metadata[satname]["filenames"]

        # load classifiers (if sklearn version above 0.20, learn the new files)
        clf, pixel_size = SDS_shoreline.load_model(satname, settings) # load the appropriate model

        # convert settings['min_beach_area'] from metres to pixels
        min_beach_area_pixels = np.ceil(settings["min_beach_area"] / pixel_size**2)

        # reduce min shoreline length for L7 because of the diagonal bands
        if satname == "L7":
            settings["min_length_sl"] = 200
        else:
            settings["min_length_sl"] = default_min_length_sl

        if satname == "L7":
            logger.info(
                f"WARNING: CoastSat has hard-coded the value for the minimum shoreline length for L7 to 200\n\n"
            )
        logger.info(
            f"Extracting shorelines for {satname} Minimum Shoreline Length: {settings['min_length_sl']}\n\n"
        )

        # loop through the images
        for i in tqdm(
            range(len(filenames)),
            desc=f"{satname}: Mapping Shorelines",
            leave=True,
            position=0,
        ):
            apply_cloud_mask = settings.get("apply_cloud_mask", True)
            # get image filename
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            shoreline_date = os.path.basename(fn[0])[:19]

            # preprocess image (cloud mask + pansharpening/downsampling)
            (
                im_ms,
                georef,
                cloud_mask,
                im_extra,
                im_QA,
                im_nodata,
            ) = SDS_preprocess.preprocess_single(
                fn,
                satname,
                settings["cloud_mask_issue"],
                settings["pan_off"],
                collection,
                apply_cloud_mask,
                settings.get("s2cloudless_prob", 60),
            )
            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]["epsg"][i]

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.sum(cloud_mask) / cloud_mask.size

            if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
                logger.error(
                    f"Skipping {satname} {shoreline_date} due to cloud & no data pixels exceeding the maximum percentage allowed: {cloud_cover_combined:.2%} > 99%\n\n"
                )
                continue

            # remove no data pixels from the cloud mask
            # (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)

            # compute updated cloud cover percentage (without no data pixels)
            valid_pixels = np.sum(~im_nodata)
            cloud_cover = np.sum(cloud_mask_adv.astype(int)) / valid_pixels.astype(int)
            # skip image if cloud cover is above user-defined threshold
            if cloud_cover > settings["cloud_thresh"]:
                logger.error(
                    f"Skipping {satname} {shoreline_date} due to cloud cover percentage exceeding cloud threshold: {cloud_cover:.2%} > {settings['cloud_thresh']:.2%}.\n\n"
                )
                continue
            else:
                logger.info(f"\nProcessing image {satname} {shoreline_date}")

            logger.info(f"{satname} {shoreline_date} cloud cover : {cloud_cover:.2%}")

            # filter out shorelines whose geoaccuracy is below the threshold
            geoacc = metadata[satname]["acc_georef"][i]
            if geoacc in ["PASSED", "FAILED"]:
                if geoacc != "PASSED":
                    continue
                else:
                    if geoacc <= geoaccuracy_threshold:
                        continue

            # calculate a buffer around the reference shoreline (if any has been digitised)
            im_ref_buffer = SDS_shoreline.create_shoreline_buffer(
                cloud_mask.shape, georef, image_epsg, pixel_size, settings
            )

            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = SDS_shoreline.classify_image_NN(
                im_ms, cloud_mask, min_beach_area_pixels, clf
            )
            # sand, whitewater, water, other
            class_mapping = {
                0: "sand",
                1: "whitewater",
                2: "water",
            }

            logger.info(
                f"{satname} {shoreline_date}: "
                + f" ,".join(
                    f"{class_name}: {np.sum(im_labels[:, :, index])/im_labels[:, :, index].size:.2%}"
                    for index, class_name in class_mapping.items()
                )
            )

            # otherwise map the contours automatically with one of the two following functions:
            # if there are pixels in the 'sand' class --> use find_wl_contours2 (enhanced)
            # otherwise use find_wl_contours1 (traditional)
            try:  # use try/except structure for long runs
                if (
                    sum(im_labels[im_ref_buffer, 0]) < 50
                ):  # minimum number of sand pixels
                    # compute MNDWI image (SWIR-G)
                    im_mndwi = SDS_tools.nd_index(
                        im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask
                    )
                    logger.info(
                        f"{satname} {shoreline_date}: Less than 50 sand pixels detected within reference shoreline buffer. Using find_wl_contours1"
                    )
                    # find water contours on MNDWI grayscale image
                    contours_mwi, t_mndwi = SDS_shoreline.find_wl_contours1(
                        im_mndwi, cloud_mask, im_ref_buffer
                    )
                else:
                    logger.info(
                        f"{satname} {shoreline_date}: Greater than 50 sand pixels detected within reference shoreline buffer. Using find_wl_contours2"
                    )
                    # use classification to refine threshold and extract the sand/water interface
                    contours_mwi, t_mndwi = SDS_shoreline.find_wl_contours2(
                        im_ms, im_labels, cloud_mask, im_ref_buffer
                    )
            except Exception as e:
                print(
                    f"{satname} {shoreline_date}: Could not map shoreline due to error {str(e)}"
                )
                logger.error(
                    f"{satname} {shoreline_date}: Could not map shoreline due to error {e}\n{traceback.format_exc()}"
                )
                continue
            date = filenames[i][:19]
            # process the water contours into a shoreline (shorelines are in the epsg of the image)
            shoreline_gdf = process_shoreline(
                contours_mwi,
                cloud_mask_adv,
                im_nodata,
                georef,
                image_epsg,
                settings,
                metadata[satname]["dates"][i],
                logger=logger,
            )
    
            # convert the polygon coordinates of ROI to gdf
            height, width = im_ms.shape[:2]
            output_epsg = settings["output_epsg"]
            roi_gdf = SDS_preprocess.create_gdf_from_image_extent(
                height, width, georef, image_epsg, output_epsg
            )
            # filter shorelines to only keep those within the extraction area
            filtered_shoreline_gdf = filter_shoreline_new(
                shoreline_gdf, shoreline_extraction_area, output_epsg
            )
 
            shoreline_extraction_area_array = (
                SDS_shoreline.get_extract_shoreline_extraction_area_array(
                    shoreline_extraction_area, output_epsg, roi_gdf
                )
            )

            # convert the shorelines to a list of numpy arrays that can be plotted
            single_shoreline = []
            for geom in filtered_shoreline_gdf.geometry:
                single_shoreline.append(np.array(geom.coords))
            shoreline_array = extract_contours(single_shoreline)

            # visualize the mapped shorelines, there are two options:
            if settings["save_figure"]:
                date = filenames[i][:19]
                SDS_shoreline.show_detection(
                    im_ms,
                    cloud_mask,
                    im_labels,
                    shoreline_array,
                    image_epsg,
                    georef,
                    settings,
                    date,
                    satname,
                    im_ref_buffer,
                    output_directory,
                    shoreline_extraction_area_array,
                )
            # if the shoreline is empty, skip it
            if len(filtered_shoreline_gdf) == 0:
                continue
    
            # append to output variables
            filtered_shoreline_gdf["filename"] = np.tile(
                filenames[i], len(filtered_shoreline_gdf)
            )
            filtered_shoreline_gdf["satname"] = np.tile(
                satname, len(filtered_shoreline_gdf)
            )
            filtered_shoreline_gdf["geoaccuracy"] = np.tile(
                metadata[satname]["acc_georef"][i], len(filtered_shoreline_gdf)
            )
            filtered_shoreline_gdf["idx"] = np.tile(i, len(filtered_shoreline_gdf))
            filtered_shoreline_gdf["MNDWI_threshold"] = np.tile(
                t_mndwi, len(filtered_shoreline_gdf)
            )
            # add the filtered shoreline to the list of all shorelines
            all_shorelines.append(filtered_shoreline_gdf)

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    # combine all the shorelines into one geodataframe
    all_shorelines_gdf = concat_and_sort_geodataframes(all_shorelines, "date")

    # convert the date column to datetime
    all_shorelines_gdf = convert_date_column_to_datetime(all_shorelines_gdf,"date")

    # save the extracted shorelines as a geodataframe to crs 4326
    # save as extracted_shorelines_lines.geojson and extracted_shorelines_points.geojson
    save_shorelines_to_geojson(all_shorelines_gdf, output_directory)

    # convert the geodataframes to a dict with one list sorted by date with all the shorelines
    shoreline_dict = aggregate_shoreline_data(all_shorelines_gdf)
    # save the extracted shorelines as a geodataframe to crs 4326
    all_shorelines_gdf_4326 = all_shorelines_gdf.to_crs("epsg:4326")
    all_shorelines_gdf_4326.to_file(
        os.path.join(filepath_data, sitename, sitename + "_shorelines.geojson"),
        driver="GeoJSON",
    )

    filepath = os.path.join(filepath_data, sitename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    json_path = os.path.join(filepath, sitename + "_output.json")
    SDS_preprocess.write_to_json(json_path, shoreline_dict)
    # release the logger as it is no longer needed
    SDS_shoreline.release_logger(logger)

    return shoreline_dict


def stringify_datetime_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Check if any of the columns in a GeoDataFrame have the type pandas timestamp and convert them to string.

    Args:
        gdf: A GeoDataFrame.

    Returns:
        A new GeoDataFrame with the same data as the original, but with any timestamp columns converted to string.
    """
    timestamp_cols = [
        col for col in gdf.columns if pd.api.types.is_datetime64_any_dtype(gdf[col])
    ]

    if not timestamp_cols:
        return gdf

    gdf = gdf.copy()

    for col in timestamp_cols:
        gdf[col] = gdf[col].astype(str)

    return gdf

def filter_shoreline_new(
    shoreline,
    shoreline_extraction_area,
    output_epsg,
):
    """Filter the shoreline based on the extraction area.
    Args:
        shoreline (array): The original shoreline data.
        shoreline_extraction_area (GeoDataFrame): The area to extract the shoreline from.
        shoreline_extraction_area (GeoDataFrame): The area to extract the shoreline from.
    Returns:
        np.array: The filtered shoreline as a numpy array of shape (n,2).
    """
    if shoreline_extraction_area is not None:
        # Ensure both the shoreline and extraction area are in the same CRS.
        shoreline_extraction_area_gdf = shoreline_extraction_area.to_crs(
            f"epsg:{output_epsg}"
        )

        if isinstance(shoreline, gpd.GeoDataFrame):
            shoreline_gdf = shoreline.to_crs(f"epsg:{output_epsg}")
        else:
            # Convert the shoreline to a GeoDataFrame.
            shoreline_gdf = SDS_shoreline.create_gdf_from_type(
                shoreline,
                "lines",
                crs=f"epsg:{output_epsg}",
            )
            if shoreline_gdf is None:
                return shoreline

        # Filter shorelines within the extraction area.
        filtered_shoreline_gdf = common.ref_poly_filter(
            shoreline_extraction_area_gdf, shoreline_gdf
        )
        return filtered_shoreline_gdf

    return shoreline

def compute_intersection_QC(
    output,
    transects,
    along_dist=25,
    min_points=3,
    max_std=15,
    max_range=30,
    min_chainage=-100,
    multiple_inter="auto",
    prc_multiple=0.1,
    use_progress_bar: bool = True,
    **kwargs,
):
    """
    More advanced function to compute the intersection between the 2D mapped shorelines
    and the transects. Produces more quality-controlled time-series of shoreline change.
    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
        transects: dict
            contains the X and Y coordinates of the transects (first and last point needed for each
            transect).
        along_dist: int (in metres)
            alongshore distance to calculate the intersection (median of points
            within this distance).
        min_points: int
            minimum number of shoreline points to calculate an intersection.
        max_std: int (in metres)
            maximum std for the shoreline points when calculating the median,
            if above this value then NaN is returned for the intersection.
        max_range: int (in metres)
            maximum range for the shoreline points when calculating the median,
            if above this value then NaN is returned for the intersection.
        min_chainage: int (in metres)
            furthest landward of the transect origin that an intersection is
            accepted, beyond this point a NaN is returned.
        multiple_inter: mode for removing outliers ('auto', 'nan', 'max').
        prc_multiple: float, optional
            percentage to use in 'auto' mode to switch from 'nan' to 'max'.
        use_progress_bar(bool,optional). Defaults to True. If true uses tqdm to display the progress for iterating through transects.
            False, means no progress bar is displayed.
        kwargs: dict
            additional keyword arguments.(used for compatibility with other functions)
    Returns:
    -----------
        cross_dist: dict
            time-series of cross-shore distance along each of the transects. These are not tidally
            corrected.
    """

    cross_dist = {}

    shorelines = output["shorelines"]
    transect_keys = list(transects.keys())
    if use_progress_bar:
        transect_keys = tqdm(
            transect_keys, desc="Computing transect shoreline intersections"
        )

    for key in transect_keys:
        std_intersect = np.full(len(shorelines), np.nan)
        med_intersect = np.full(len(shorelines), np.nan)
        max_intersect = np.full(len(shorelines), np.nan)
        min_intersect = np.full(len(shorelines), np.nan)
        n_intersect = np.full(len(shorelines), np.nan)

        transect_start = transects[key][0, :]
        transect_end = transects[key][-1, :]
        transect_vector = transect_end - transect_start
        transect_length = np.linalg.norm(transect_vector)
        transect_unit_vector = transect_vector / transect_length
        rotation_matrix = np.array(
            [
                [transect_unit_vector[0], transect_unit_vector[1]],
                [-transect_unit_vector[1], transect_unit_vector[0]],
            ]
        )

        for i, shoreline in enumerate(shorelines):
            if len(shoreline) == 0:
                continue

            shoreline_shifted = shoreline - transect_start
            shoreline_rotated = np.dot(rotation_matrix, shoreline_shifted.T).T

            d_line = np.abs(shoreline_rotated[:, 1])
            d_origin = np.linalg.norm(shoreline_shifted, axis=1)
            idx_close = (d_line <= along_dist) & (d_origin <= 1000)

            if not np.any(idx_close):
                continue

            valid_points = shoreline_rotated[idx_close, 0]
            valid_points = valid_points[valid_points >= min_chainage]

            if np.sum(~np.isnan(valid_points)) < min_points:
                continue

            std_intersect[i] = np.nanstd(valid_points)
            med_intersect[i] = np.nanmedian(valid_points)
            max_intersect[i] = np.nanmax(valid_points)
            min_intersect[i] = np.nanmin(valid_points)
            n_intersect[i] = np.sum(~np.isnan(valid_points))

        condition1 = std_intersect <= max_std
        condition2 = (max_intersect - min_intersect) <= max_range
        condition3 = n_intersect >= min_points
        idx_good = condition1 & condition2 & condition3

        if multiple_inter == "auto":
            prc_over = np.sum(std_intersect > max_std) / len(std_intersect)
            if prc_over > prc_multiple:
                med_intersect[~idx_good] = max_intersect[~idx_good]
                med_intersect[~condition3] = np.nan
            else:
                med_intersect[~idx_good] = np.nan
        elif multiple_inter == "max":
            med_intersect[~idx_good] = max_intersect[~idx_good]
            med_intersect[~condition3] = np.nan
        elif multiple_inter == "nan":
            med_intersect[~idx_good] = np.nan
        else:
            raise ValueError(
                "The multiple_inter parameter can only be: nan, max, or auto."
            )

        cross_dist[key] = med_intersect

    return cross_dist

def filter_contours_by_length(
    contours_epsg: list[np.ndarray], min_length_sl: float
) -> list[np.ndarray]:
    """
    Filters contours by their length.
    Args:
        contours_epsg (list[np.ndarray]): List of contours, where each contour is an array of coordinates.
        min_length_sl (float): Minimum length threshold for the contours.
    Returns:
        list[np.ndarray]: List of contours that meet the minimum length requirement.
    """
    contours_long = []
    for wl in contours_epsg:
        coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
        a = LineString(coords)
        if a.length >= min_length_sl:
            contours_long.append(wl)
    return contours_long


def filter_points_within_distance_to_mask(
    contours_2d: list[np.ndarray],
    mask: np.ndarray,
    georef: np.ndarray,
    image_epsg: int,
    output_epsg: int,
    distance_threshold: float = 60,
) -> list[np.ndarray]:
    """
    Filters points within a specified distance to a mask.
    Args:
        contours_2d (list[np.ndarray]): List of contours, where each contour is an array of coordinates.
        mask (np.ndarray): Binary mask array.
        georef (np.ndarray): Georeference information.
        image_epsg (int): EPSG code of the image coordinate system.
        output_epsg (int): EPSG code of the output coordinate system.
        distance_threshold (float, optional): Distance threshold for filtering. Defaults to 60.
    Returns:
        list[np.ndarray]: List of contours filtered by the distance to the mask.
    """
    idx_mask = np.where(mask)
    idx_mask = np.array(
        [(idx_mask[0][k], idx_mask[1][k]) for k in range(len(idx_mask[0]))]
    )
    if len(idx_mask) == 0:
        return contours_2d
    coords_in_epsg = SDS_tools.convert_epsg(
        SDS_tools.convert_pix2world(idx_mask, georef), image_epsg, output_epsg
    )[:, :-1]
    coords_tree = KDTree(coords_in_epsg)
    new_contours = filter_shorelines_by_distance(
        contours_2d, coords_tree, distance_threshold
    )
    return new_contours


def extract_contours(filtered_contours_long: list[np.ndarray]):
    """
    Extracts x and y coordinates from a list of contours and combines them into a single array.
    Args:
        filtered_contours_long (list): List of contours, where each contour is a numpy array with at least 2 columns.
    Returns:
        np.ndarray: A transposed array with x coordinates in the first column and y coordinates in the second column.
    """
    only_points = [contour[:, :2] for contour in filtered_contours_long]
    x_points = np.array([])
    y_points = np.array([])

    for points in only_points:
        x_points = np.append(x_points, points[:, 0])
        y_points = np.append(y_points, points[:, 1])

    contours_array = np.transpose(np.array([x_points, y_points]))
    return contours_array

def filter_shorelines_by_distance(
    contours_2d: list[np.ndarray], coords_tree: KDTree, distance_threshold: float = 60
) -> list[np.ndarray]:
    """
    Filters shorelines by their distance to a set of coordinates.
    Args:
        contours_2d (list[np.ndarray]): List of contours, where each contour is an array of coordinates.
        coords_tree (KDTree): KDTree of coordinates to compare distances against.
        distance_threshold (float, optional): Distance threshold for filtering. Defaults to 60.
    Returns:
        list[np.ndarray]: List of filtered shorelines.
    """
    new_contours = []
    for shoreline in contours_2d:
        distances, _ = coords_tree.query(
            shoreline, distance_upper_bound=distance_threshold
        )
        idx_keep = distances >= distance_threshold
        new_shoreline = shoreline[idx_keep]
        if len(new_shoreline) > 0:
            new_contours.append(new_shoreline)
    return new_contours

def check_percent_no_data_allowed(
    percent_no_data_allowed: float, cloud_mask: np.ndarray, im_nodata: np.ndarray
) -> bool:
    """
    Checks if the percentage of no data pixels in the image exceeds the allowed percentage.

    Args:
        settings (dict): A dictionary containing settings for the shoreline extraction.
        cloud_mask (numpy.ndarray): A binary mask indicating cloud cover in the image.
        im_nodata (numpy.ndarray): A binary mask indicating no data pixels in the image.

    Returns:
        bool: True if the percentage of no data pixels is less than or equal to the allowed percentage, False otherwise.
    """
    if percent_no_data_allowed is not None:
        num_total_pixels = cloud_mask.shape[0] * cloud_mask.shape[1]
        percentage_no_data = np.sum(im_nodata) / num_total_pixels
        if percentage_no_data > percent_no_data_allowed:
            logger.info(
                f"percent_no_data_allowed exceeded {percentage_no_data} > {percent_no_data_allowed}"
            )
            return False
    return True


def convert_linestrings_to_multipoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert LineString geometries in a GeoDataFrame to MultiPoint geometries.

    Args:
    - gdf (gpd.GeoDataFrame): The input GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: A new GeoDataFrame with MultiPoint geometries. If the input GeoDataFrame
                        already contains MultiPoints, the original GeoDataFrame is returned.
    """

    # Check if the gdf already contains MultiPoints
    if any(gdf.geometry.type == "MultiPoint"):
        return gdf

    def linestring_to_multipoint(linestring):
        if isinstance(linestring, LineString):
            return MultiPoint(linestring.coords)
        return linestring

    # Convert each LineString to a MultiPoint
    gdf["geometry"] = gdf["geometry"].apply(linestring_to_multipoint)

    return gdf


def transform_gdf_to_crs(gdf, crs=4326):
    """Convert the GeoDataFrame to the specified CRS."""
    return gdf.to_crs(crs)


def select_and_stringify(gdf, row_number):
    """Select a single shoreline and stringify its datetime columns."""
    single_shoreline = gdf.iloc[[row_number]]
    return common.stringify_datetime_columns(single_shoreline)


def convert_gdf_to_json(gdf):
    """Convert a GeoDataFrame to a JSON representation."""
    return json.loads(gdf.to_json())


def style_layer(
    geojson: dict, layer_name: str, color: str, style_dict: dict = {}
) -> GeoJSON:
    """Return styled GeoJson object with layer name
    Args:
        geojson (dict): geojson dictionary to be styled
        layer_name(str): name of the GeoJSON layer
        color(str): hex code or name of color render shorelines
        style_dict (dict, optional): Additional style attributes to be merged with the default style.
    Returns:
        "ipyleaflet.GeoJSON": shoreline as GeoJSON layer styled with color
    """
    assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
    # Default style dictionary
    default_style = {
        "color": color,  # Outline color
        "opacity": 1,  # opacity 1 means no transparency
        "weight": 3,  # Width
        "fillColor": color,  # Fill color
        "fillOpacity": 0.8,  # Fill opacity.
        "radius": 1,
    }

    # If a style_dict is provided, merge it with the default style
    default_style.update(style_dict)
    return GeoJSON(
        data=geojson, name=layer_name, style=default_style, point_style=default_style
    )


def read_from_dict(d: dict, keys_of_interest: list | set | tuple):
    """
    Function to extract the value from the first matching key in a dictionary.

    Parameters:
    d (dict): The dictionary from which to extract the value.
    keys_of_interest (list | set | tuple): Iterable of keys to look for in the dictionary.
    The function returns the value of the first matching key it finds.

    Returns:
    The value from the dictionary corresponding to the first key found in keys_of_interest,
    or None if no matching keys are found.
    Raises:
    KeyError if the keys_of_interest were not in d
    """
    for key in keys_of_interest:
        if key in d:
            return d[key]
    raise KeyError(f"{keys_of_interest} were not in {d}")


def remove_small_objects_and_binarize(merged_labels, min_size):
    # Ensure the image is binary
    binary_image = merged_labels > 0

    # Remove small objects from the binary image
    filtered_image = morphology.remove_small_objects(
        binary_image, min_size=min_size, connectivity=2
    )

    return filtered_image


def compute_transects_from_roi(
    extracted_shorelines: dict,
    transects_gdf: gpd.GeoDataFrame,
    settings: dict,
) -> dict:
    """Computes the intersection between the 2D shorelines and the shore-normal.
        transects. It returns time-series of cross-shore distance along each transect.
    Args:
        extracted_shorelines (dict): contains the extracted shorelines and corresponding metadata
        transects_gdf (gpd.GeoDataFrame): transects in ROI with crs = output_crs in settings
        settings (dict): settings dict with keys
                    'along_dist': int
                        alongshore distance considered calculate the intersection
    Returns:
        dict:  time-series of cross-shore distance along each of the transects.
               Not tidally corrected.
    """
    # create dict of numpy arrays of transect start and end points

    transects = common.get_transect_points_dict(transects_gdf)
    # cross_distance: along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
    cross_distance = compute_intersection_QC(extracted_shorelines, transects, **settings)
    return cross_distance


def combine_satellite_data(satellite_data: dict) -> dict:
    """
    Function to merge the satellite_data dictionary, which has one key per satellite mission
    into a dictionnary containing all the shorelines and dates ordered chronologically.

    Arguments:
    -----------
    satellite_data: dict
        contains the extracted shorelines and corresponding dates, organised by
        satellite mission

    Returns:
    -----------
    merged_satelllite_data: dict
        contains the extracted shorelines in a single list sorted by date

    """
    # Initialize merged_satellite_data dict
    merged_satellite_data = {
        "dates": [],
        "geoaccuracy": [],
        "shorelines": [],
        "idx": [],
        "satname": [],
    }

    # Iterate through satellite_data keys (satellite names)
    for satname in satellite_data:
        # Iterate through each key in the nested dictionary
        for key in satellite_data[satname].keys():
            # Add the key to merged_satellite_data if it doesn't already exist
            if key not in merged_satellite_data:
                merged_satellite_data[key] = []

    # Add an additional key for the satellite name
    merged_satellite_data["satname"] = []

    # Fill the satellite_data dict
    for satname, sat_data in satellite_data.items():
        satellite_data[satname].setdefault("dates", [])
        satellite_data[satname].setdefault("geoaccuracy", [])
        satellite_data[satname].setdefault("shorelines", [])
        satellite_data[satname].setdefault("cloud_cover", [])
        satellite_data[satname].setdefault("filename", [])
        satellite_data[satname].setdefault("idx", [])
        # For each key in the nested dictionary
        for key, value in sat_data.items():
            # Wrap non-list values in a list and concatenate to merged_satellite_data
            if not isinstance(value, list):
                merged_satellite_data[key] += [value]
            else:
                merged_satellite_data[key] += value
            # Add the satellite name to the satellite name list
        if "dates" in satellite_data[satname].keys():
            merged_satellite_data["satname"] += [
                _ for _ in np.tile(satname, len(satellite_data[satname]["dates"]))
            ]
    # Sort dates chronologically
    if "dates" in merged_satellite_data.keys():
        idx_sorted = sorted(
            range(len(merged_satellite_data["dates"])),
            key=lambda i: merged_satellite_data["dates"][i],
        )
        for key in merged_satellite_data.keys():
            merged_satellite_data[key] = [
                merged_satellite_data[key][i] for i in idx_sorted
            ]

    return merged_satellite_data

def LineString_to_arr(line):
    """
    Makes an array from linestring
    inputs: line
    outputs: array of xy tuples
    """
    listarray = []
    for pp in line.coords:
        listarray.append(pp)
    nparray = np.array(listarray)
    return nparray

def arr_to_LineString(coords):
    """
    Makes a line feature from a list of xy tuples
    inputs: coords
    outputs: line
    """
    points = [None]*len(coords)
    i=0
    for xy in coords:
        points[i] = shapely.geometry.Point(xy)
        i=i+1
    line = shapely.geometry.LineString(points)
    return line

def chaikins_corner_cutting(coords, refinements=5):
    """
    Smooths out lines or polygons with Chaikin's method
    """
    i=0
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25
        i=i+1
    return coords

def smooth_lines(lines:gpd.GeoDataFrame,refinements=5):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM
    saves output with '_smooth' appended to original filename in same directory

    inputs:
    lines (gpd.GeoDataFrame): GeoDataFrame containing shorelines in UTM
    refinements (int): number of refinemnets for Chaikin's smoothing algorithm
    outputs:
    save_path (str): path of output file in UTM
    """
    new_lines = lines.copy()
    for i in range(len(lines)):
        line = lines.iloc[i]
        coords = LineString_to_arr(line.geometry)
        refined = chaikins_corner_cutting(coords, refinements=refinements)
        refined_geom = arr_to_LineString(refined)
        new_lines.loc[i,'geometry'] = refined_geom
    return new_lines

def process_shoreline_zoo(
    contours, cloud_mask, im_nodata, georef, image_epsg, settings, date,satname:str,**kwargs
):
    # convert the contours that are currently pixel coordinates to world coordiantes
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    contours_epsg = SDS_tools.convert_epsg(
        contours_world, image_epsg, settings["output_epsg"]
    )
    # this is the shoreline in the form of a list of numpy arrays, each array containing the coordinates of a shoreline x,y,z
    contours_long = filter_contours_by_length(contours_epsg, settings["min_length_sl"])
    # this removes the z coordinate from each shoreline point, so the format is list of numpy arrays, each array containing the x,y coordinates of a shoreline point
    contours_2d = [contour[:, :2] for contour in contours_long]
    # remove shoreline points that are too close to the no data mask
    new_contours = filter_points_within_distance_to_mask(
        contours_2d,
        im_nodata,
        georef,
        image_epsg,
        settings["output_epsg"],
        distance_threshold=60,
    )
    # remove shoreline points that are too close to the cloud mask
    new_contours = filter_points_within_distance_to_mask(
        new_contours,
        cloud_mask,
        georef,
        image_epsg,
        settings["output_epsg"],
        distance_threshold=settings["dist_clouds"],
    )
    filtered_contours_long = filter_contours_by_length(
        new_contours, settings["min_length_sl"]
    )
    contours_shapely = [LineString(contour) for contour in filtered_contours_long]
    if isinstance(date, str):
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d-%H-%M-%S")
    else:
        date_obj = date

    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
    # compute updated cloud cover percentage (without no data pixels)
    valid_pixels = np.sum(~im_nodata)
    cloud_cover = np.sum(cloud_mask_adv.astype(int)) / valid_pixels.astype(int)

    gdf = gpd.GeoDataFrame(
        {
            "date": np.tile(date_obj, len(contours_shapely)), # type: ignore
            "satname": np.tile(satname, len(contours_shapely)), # type: ignore
            "cloud_cover": np.tile(cloud_cover, len(contours_shapely)),
        },
        geometry=contours_shapely,
        crs=f"EPSG:{image_epsg}",
    )

    # smooth the shorelines in the GeoDataFrame
    gdf = smooth_lines(gdf)
    return gdf

def find_shoreline(
    filename: str,
    image_epsg: int,
    settings: dict,
    cloud_mask_adv: np.ndarray,
    cloud_mask: np.ndarray,
    im_nodata: np.ndarray,
    georef: float,
    im_labels: np.ndarray,
    reference_shoreline_buffer: np.ndarray,
    date: str,
    satname: str,
) -> np.array:
    """
    Finds the shoreline in an image.
    Args:
        fn (str): The filename of the image.
        image_epsg (int): The EPSG code of the image.
        settings (dict): A dictionary containing settings for the shoreline extraction.
        cloud_mask_adv (numpy.ndarray): A binary mask indicating advanced cloud cover in the image.
        cloud_mask (numpy.ndarray): A binary mask indicating cloud cover in the image.
        im_nodata (numpy.ndarray): A binary mask indicating no data pixels in the image.
        georef (flat): A the georeference code for the image.
        im_labels (numpy.ndarray): A labeled array indicating the water and land pixels in the image.
        reference_shoreline_buffer (numpy.ndarray,): A buffer around the reference shoreline.
    Returns:
        numpy.ndarray or None: The shoreline as a numpy array, or None if the shoreline could not be found.
    """
    try:
        contours = simplified_find_contours(
            im_labels, cloud_mask, reference_shoreline_buffer
        )
    except Exception as e:
        logger.error(f"{e}\nCould not map shoreline for this image: {filename}")
        return None
    shoreline = process_shoreline_zoo(
        contours, cloud_mask_adv, im_nodata, georef, image_epsg, settings,date,satname,
    )
    # this is a geodataframe with the shoreline in it with the date and cloud cover
    return shoreline

def process_satellite(
    satname: str,
    settings: dict,
    metadata: dict,
    session_path: str,
    class_indices: List[int] = None,
    class_mapping: Dict[int, str] = None,
    save_location: str = "",
    batch_size: int = 10,
    shoreline_extraction_area: gpd.GeoDataFrame = None,
    **kwargs: dict,
):
    """
    Processes a satellite's imagery to extract shorelines.

    Args:
        satname (str): The name of the satellite.
        settings (dict): A dictionary containing settings for the shoreline extraction.
            Settings needed to extract shorelines
            Must contain the following keys
            'min_length_sl': int
                minimum length of shoreline to be considered
            'min_beach_area': int
                minimum area of beach to be considered
            'cloud_thresh': float
                maximum cloud cover allowed
            'cloud_mask_issue': bool
                whether to apply the cloud mask or not
            'along_dist': int
                alongshore distance considered calculate the intersection
        metadata (dict): A dictionary containing metadata for the satellite imagery.
            Metadata is the output of the get_metadata function in SDS_download.py.
            The metadata dictionary should have the following structure:
            ex.
            metadata = {
                "l8": {
                    "dates": ["2019-01-01", "2019-01-02"],
                    "filenames": ["2019-01-01_123456789.tif", "2019-01-02_123456789.tif", "2019-01-03_123456789.tif"],
                    "epsg": [32601, 32601, 32601],
                    "acc_georef": [True, True, True]
                },
        session_path (str): The path to the session directory.
        class_indices (list, optional): A list of class indices to extract. Defaults to None.
        class_mapping (dict, optional): A dictionary mapping class indices to class names. Defaults to None.
        save_location (str, optional): The path to save the extracted shorelines. Defaults to "".
        batch_size (int, optional): The number of images to process in each batch. Defaults to 10.
        shoreline_extraction_area (gpd.GeoDataFrame, optional): A GeoDataFrame containing the extraction area for the shorelines. Defaults to None.
    Returns:
        dict: A dictionary containing the extracted shorelines for the satellite.
    """
    # filenames of tifs (ms) for this satellite
    filenames = metadata[satname]["filenames"]
    output = {}
    gdf_list = []
    all_shorelines_gdf = gpd.GeoDataFrame()
    output.setdefault(satname, {})
    output[satname].setdefault("dates", [])
    output[satname].setdefault("geoaccuracy", [])
    output[satname].setdefault("shorelines", [])
    output[satname].setdefault("cloud_cover", [])
    output[satname].setdefault("filename", [])
    output[satname].setdefault("idx", [])

    if len(filenames) == 0:
        logger.warning(f"Satellite {satname} had no imagery")
        return output

    collection = settings["inputs"]["landsat_collection"]
    # deep copy settings
    settings = copy.deepcopy(settings)
    filepath = get_filepath(settings["inputs"], satname)
    pixel_size = get_pixel_size_for_satellite(satname)

    # get the minimum beach area in number of pixels depending on the satellite
    settings["min_length_sl"] = get_min_shoreline_length(
        satname, settings["min_length_sl"]
    )

    # loop through the images
    espg_list = []
    geoaccuracy_list = []
    timestamps = []
    tasks = []

    # compute number of batches
    num_batches = len(filenames) // batch_size
    if len(filenames) % batch_size != 0:
        num_batches += 1

    # initialize progress bar
    pbar = tqdm(
        total=len(filenames),
        desc=f"Mapping Shorelines for {satname}",
        leave=True,
        position=0,
    )

    for batch in range(num_batches):
        espg_list = []
        geoaccuracy_list = []
        timestamps = []
        tasks = []

        # generate tasks for the current batch
        for index in range(
            batch * batch_size, min((batch + 1) * batch_size, len(filenames))
        ):
            image_epsg = metadata[satname]["epsg"][index]
            # espg_list.append(image_epsg)
            geoaccuracy_list.append(metadata[satname]["acc_georef"][index])
            timestamps.append(metadata[satname]["dates"][index])
            tasks.append(
                dask.delayed(process_satellite_image)(
                    filenames[index],
                    filepath,
                    settings,
                    satname,
                    metadata[satname]["dates"][index],
                    collection,
                    image_epsg,
                    pixel_size,
                    session_path,
                    class_indices,
                    class_mapping,
                    save_location,
                    settings.get("apply_cloud_mask", True),
                    shoreline_extraction_area,
                    index = index,
                    geoaccuracy = metadata[satname]["acc_georef"][index],
                )
            )

        # compute tasks in batches
        results = dask.compute(*tasks)
        # update progress bar
        num_tasks_computed = len(tasks)
        pbar.update(num_tasks_computed)

        # merge resulting geodataframes
        new_gdf_list = [result for result in results if result is not None and isinstance(result, gpd.GeoDataFrame)]
        gdf_list.extend(new_gdf_list)
        all_shorelines_gdf = concat_and_sort_geodataframes(gdf_list, "date", "UTC")
    pbar.close()
    # return output
    return all_shorelines_gdf


def get_cloud_cover_combined(cloud_mask: np.ndarray):
    """
    Calculate the cloud cover percentage of a cloud_mask.
    Note: The way that cloud_mask is created in SDS_preprocess.preprocess_single() means that it will contain 1's where no data pixels were detected.
    TLDR the cloud mask is combined with the no data mask. No idea why.

    Parameters:
    cloud_mask (numpy.ndarray): A 2D numpy array with 0s (clear) and 1s (cloudy) representing the cloud mask.

    Returns:
    float: The percentage of cloud_cover_combined in the cloud_mask.
    """
    # Convert cloud_mask to integer and calculate the sum of all elements (number of cloudy pixels)
    num_cloudy_pixels = sum(sum(cloud_mask.astype(int)))

    # Calculate the total number of pixels in the cloud_mask
    num_total_pixels = cloud_mask.shape[0] * cloud_mask.shape[1]

    # Divide the number of cloudy pixels by the total number of pixels to get the cloud_cover_combined percentage
    cloud_cover_combined = np.divide(num_cloudy_pixels, num_total_pixels)

    return cloud_cover_combined


def get_cloud_cover(cloud_mask: np.ndarray, im_nodata: np.ndarray) -> float:
    """
    Calculate the cloud cover percentage in an image, considering only valid (non-no-data) pixels.

    Args:
    cloud_mask (numpy.array): A boolean 2D numpy array where True represents a cloud pixel,
        and False a non-cloud pixel.
    im_nodata (numpy.array): A boolean 2D numpy array where True represents a no-data pixel,
        and False a valid (non-no-data) pixel.

    Returns:
    float: The cloud cover percentage in the image (0-1), considering only valid (non-no-data) pixels.
    """

    # Remove no data pixels from the cloud mask, as they should not be included in the cloud cover calculation
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)

    # Compute updated cloud cover percentage without considering no data pixels
    cloud_cover = np.divide(
        sum(sum(cloud_mask_adv.astype(int))),
        (sum(sum((~im_nodata).astype(int)))),
    )

    return cloud_cover

def concat_and_sort_geodataframes(
    gdfs: list[gpd.GeoDataFrame], date_column: str, timezone: str = "UTC"
) -> gpd.GeoDataFrame:
    """
    Concatenates a list of GeoDataFrames with the same columns into a single GeoDataFrame and sorts by a date column.

    Args:
        gdfs (list[gpd.GeoDataFrame]): List of GeoDataFrames to concatenate.
        date_column (str): The name of the date column to sort by.
        timezone (str): The timezone to which naive datetime entries should be localized. Default is 'UTC'.

    Returns:
        gpd.GeoDataFrame: A single concatenated and sorted GeoDataFrame.
    
    """
    sorted_gdf = gpd.GeoDataFrame()
    if gdfs is None or len(gdfs) == 0:
        print("No GeoDataFrames to concatenate")
        return sorted_gdf
    concatenated_gdf = pd.concat(gdfs, ignore_index=True)
    concatenated_gdf = gpd.GeoDataFrame(concatenated_gdf)

    # Ensure the date column is in datetime format and remove any NaT values
    concatenated_gdf[date_column] = pd.to_datetime(
        concatenated_gdf[date_column], errors="coerce"
    )
    concatenated_gdf = concatenated_gdf.dropna(subset=[date_column])
    tz = pytz.timezone(timezone)

    # Localize timezone-naive datetimes to the specified timezone
    concatenated_gdf[date_column] = concatenated_gdf[date_column].apply(
        lambda x: x.tz_localize('UTC').tz_convert(timezone) if x.tzinfo is None else x.tz_convert(timezone)
    )
    # Define timezone-aware min and max dates
    min_date = pd.Timestamp.min.tz_localize('UTC').tz_convert(tz)
    max_date = pd.Timestamp.max.tz_localize('UTC').tz_convert(tz)

    # Filter out-of-bounds datetime values
    concatenated_gdf = concatenated_gdf[
        (concatenated_gdf[date_column] > min_date)
        & (concatenated_gdf[date_column] < max_date)
    ]

    sorted_gdf = concatenated_gdf.sort_values(by=date_column).reset_index(drop=True)

    # Format the date column to the desired string format
    sorted_gdf[date_column] = sorted_gdf[date_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    return sorted_gdf

def process_satellite_image(
    filename: str,
    filepath: str,
    settings: Dict[str, Dict[str, Union[str, int, float]]],
    satname: str,
    date:str,
    collection: str,
    image_epsg: int,
    pixel_size: float,
    session_path: str,
    class_indices: List[int] = None,
    class_mapping: Dict[int, str] = None,
    save_location: str = "",
    apply_cloud_mask: bool = True,
    shoreline_extraction_area : gpd.GeoDataFrame = None,
    index: int = None,
    geoaccuracy: str = None,
) -> gpd.GeoDataFrame:
    """
    Processes a single satellite image to extract the shoreline.

    Args:
        filename (str): The filename of the image.
        filepath (str): The path to the directory containing the image.
        settings (dict): A dictionary containing settings for the shoreline extraction.
        satname (str): The name of the satellite.
        collection (str): The name of the Landsat collection.
        image_epsg (int): The EPSG code of the image.
        pixel_size (float): The pixel size of the image.
        session_path (str): The path to the session directory.
        class_indices (list, optional): A list of class indices to extract. Defaults to None.
        class_mapping (dict, optional): A dictionary mapping class indices to class names. Defaults to None.
        save_location (str, optional): The path to save the extracted shorelines. Defaults to "".
        apply_cloud_mask (bool, optional): Whether to apply the cloud mask. Defaults to True.

    Returns:
        dict: A dictionary containing the extracted shoreline and cloud cover percentage.
    """
    # get image date
    date = filename[:19]
    # get the filenames for each of the tif files (ms, pan, qa)
    fn = get_filenames(filename, filepath, satname)
    # preprocess image (cloud mask + pansharpening/downsampling)
    (
        im_ms,
        georef,
        cloud_mask,
        im_extra,
        im_QA,
        im_nodata,
    ) = SDS_preprocess.preprocess_single(
        fn,
        satname,
        settings.get("cloud_mask_issue", False),
        False,
        collection,
        do_cloud_mask=apply_cloud_mask,
    )
    # if percentage of no data pixels are greater than allowed, skip
    percent_no_data_allowed = settings.get("percent_no_data", None)
    if not check_percent_no_data_allowed(
        percent_no_data_allowed, cloud_mask, im_nodata
    ):
        logger.info(
            f"percent_no_data_allowed > {settings.get('percent_no_data', None)}: {filename}"
        )
        return None

    # compute cloud_cover percentage (with no data pixels)
    cloud_cover_combined = get_cloud_cover_combined(cloud_mask)
    if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
        logger.info(f"cloud_cover_combined > 0.99 : {filename} ")
        return None

    # compute cloud cover percentage (without no data pixels)
    cloud_cover = get_cloud_cover(cloud_mask, im_nodata)
    # skip image if cloud cover is above user-defined threshold
    if cloud_cover > settings["cloud_thresh"]:
        logger.info(f"Cloud thresh exceeded for {filename}")
        return None
    # calculate a buffer around the reference shoreline (if any has been digitised)
    # buffer is dilated by 5 pixels
    ref_shoreline_buffer = SDS_shoreline.create_shoreline_buffer(
        cloud_mask.shape, georef, image_epsg, pixel_size, settings
    )
    # read the model outputs from the npz file for this image
    npz_file = find_matching_npz(filename, os.path.join(session_path, "good"))
    if npz_file is None:
        npz_file = find_matching_npz(filename, session_path)
    # logger.info(f"npz_file: {npz_file}")
    if npz_file is None:
        logger.warning(f"npz file not found for {filename}")
        return None

    # get the labels for water and land
    land_mask = load_merged_image_labels(npz_file, class_indices=class_indices)
    all_labels = load_image_labels(npz_file)

    min_beach_area = settings["min_beach_area"]
    land_mask = remove_small_objects_and_binarize(land_mask, min_beach_area)


    # get the shoreline from the image
    shoreline = find_shoreline(
        fn,
        image_epsg,
        settings,
        np.logical_xor(cloud_mask, im_nodata),
        cloud_mask,
        im_nodata,
        georef,
        land_mask,
        ref_shoreline_buffer,
        date = date,
        satname=satname,
    )
    if shoreline is None:
        logger.warning(f"\nShoreline not found for {fn}")
        return None
    
    # convert the polygon coordinates of ROI to gdf
    height,width=im_ms.shape[:2]
    output_epsg = settings["output_epsg"]
    roi_gdf = SDS_preprocess.create_gdf_from_image_extent(height,width, georef,image_epsg,output_epsg)
    # filter shorelines within the extraction area
    shoreline = filter_shoreline_new(shoreline,shoreline_extraction_area,output_epsg)

    shoreline_extraction_area_array = SDS_shoreline.get_extract_shoreline_extraction_area_array(shoreline_extraction_area, output_epsg, roi_gdf)
    
    single_shoreline = []
    for geom in shoreline.geometry:
        single_shoreline.append(np.array(geom.coords))
    # convert the shoreline gdf to a numpy array where all the shoreline points are consolidated into a single array
    shoreline_array = extract_contours(single_shoreline)

    # plot the results
    shoreline_detection_figures(
        im_ms,
        cloud_mask,
        land_mask,
        all_labels,
        shoreline_array,
        image_epsg,
        georef,
        settings,
        date,
        satname,
        class_mapping,
        save_location,
        ref_shoreline_buffer,
        shoreline_extraction_area=shoreline_extraction_area_array,
    )
    shoreline["filename"] = np.tile(filename, len(shoreline))
    shoreline["idx"] = np.tile(index, len(shoreline))
    shoreline["geoaccuracy"] = np.tile(geoaccuracy, len(shoreline))
    return shoreline


def get_model_card_classes(model_card_path: str) -> dict:
    """return the classes dictionary from the model card
        example classes dictionary {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    Args:
        model_card_path (str): path to model card

    Returns:
        dict: dictionary of classes in model card and their corresponding index
    """
    model_card_data = file_utilities.read_json_file(model_card_path, raise_error=True)
    # logger.info(
    #     f"model_card_path: {model_card_path} \nmodel_card_data: {model_card_data}"
    # )
    # read the classes the model was trained with from either the dictionary under key "DATASET" or "DATASET1"
    model_card_dataset = common.get_value_by_key_pattern(
        model_card_data, patterns=("DATASET", "DATASET1")
    )
    model_card_classes = model_card_dataset["CLASSES"]
    return model_card_classes


def get_class_mapping(
    model_card_path: str,
) -> dict:
    # example dictionary {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    model_card_classes = get_model_card_classes(model_card_path)

    class_mapping = {}
    # get index of each class in class_mapping to match model card classes
    for index, class_name in model_card_classes.items():
        class_mapping[index] = class_name
    # return list of indexes of selected_class_names that were found in model_card_classes
    return class_mapping


def get_indices_of_classnames(
    model_card_path: str,
    selected_class_names: List[str],
) -> List[int]:
    """
    Given the path to a model card and a list of selected class names, returns a list of indices of the selected classes
    in the model card. The model card should be a dictionary that maps class indices to class names.

    :param model_card_path: a string specifying the path to the model card.
    :param selected_class_names: a list of strings specifying the names of the selected classes.
    :return: a list of integers specifying the indices of the selected classes in the model card.
    """
    # example dictionary {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    model_card_classes = get_model_card_classes(model_card_path)

    class_indices = []
    # get index of each class in class_mapping to match model card classes
    for index, class_name in model_card_classes.items():
        # see if the class name is in selected_class_names
        for selected_class_name in selected_class_names:
            if class_name == selected_class_name:
                class_indices.append(int(index))
                break
    # return list of indexes of selected_class_names that were found in model_card_classes
    return class_indices


def find_matching_npz(filename, directory):
    # Extract the timestamp and Landsat ID from the filename
    parts = filename.split("_")
    timestamp, landsat_id = parts[0], parts[1]
    # Construct a pattern to match the corresponding npz filename
    pattern = f"{timestamp}*{landsat_id}*.npz"

    # Search the directory for files that match the pattern
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, pattern):
            return os.path.join(directory, file)

    # If no matching file is found, return None
    return None


def merge_classes(im_labels: np.ndarray, classes_to_merge: list) -> np.ndarray:
    """
    Merge the specified classes in the given numpy array of class labels by creating a new numpy array with 1 values
    for the merged classes and 0 values for all other classes.

    :param im_labels: a numpy array of class labels.
    :param classes_to_merge: a list of class labels to merge.
    :return: an integer numpy array with 1 values for the merged classes and 0 values for all other classes.
    """
    # Create an integer numpy array with 1 values for the merged classes and 0 values for all other classes
    updated_labels = np.zeros(shape=(im_labels.shape[0], im_labels.shape[1]), dtype=int)

    # Set 1 values for merged classes
    for idx in classes_to_merge:
        updated_labels = np.logical_or(updated_labels, im_labels == idx).astype(int)

    return updated_labels


def load_image_labels(npz_file: str) -> np.ndarray:
    """
    Load in image labels from a .npz file. Loads in the "grey_label" array from the .npz file and returns it as a 2D

    Parameters:
    npz_file (str): The path to the .npz file containing the image labels.

    Returns:
    np.ndarray: A 2D numpy array containing the image labels from the .npz file.
    """
    if not os.path.isfile(npz_file) or not npz_file.endswith(".npz"):
        raise ValueError(f"{npz_file} is not a valid .npz file.")

    data = np.load(npz_file)
    return data["grey_label"]


def load_merged_image_labels(
    npz_file: str, class_indices: list = [2, 1, 0]
) -> np.ndarray:
    """
    Load and process image labels from a .npz file.
    Pass in the indexes of the classes to merge. For instance, if you want to merge the water and white water classes, and
    the indexes of water is 0 and white water is 1, pass in [0, 1] as the class_indices parameter.

    Parameters:
    npz_file (str): The path to the .npz file containing the image labels.
    class_indices (list): The indexes of the classes to merge.

    Returns:
    np.ndarray: A 2D numpy array containing the image labels as 1 for the merged classes and 0 for all other classes.
    """
    if not os.path.isfile(npz_file) or not npz_file.endswith(".npz"):
        raise ValueError(f"{npz_file} is not a valid .npz file.")

    data = np.load(npz_file)
    # 1 for water, 0 for anything else (land, other, sand, etc.)
    im_labels = merge_classes(data["grey_label"], class_indices)

    return im_labels


def increase_image_intensity(
    im_ms: np.ndarray, cloud_mask: np.ndarray, prob_high: float = 99.9
) -> "np.ndarray[float]":
    """
    Increases the intensity of an image using rescale_image_intensity function from SDS_preprocess module.

    Args:
    im_ms (numpy.ndarray): Input multispectral image with shape (M, N, C), where M is the number of rows,
                         N is the number of columns, and C is the number of channels.
    cloud_mask (numpy.ndarray): A 2D binary cloud mask array with the same dimensions as the input image. The mask should have True values where cloud pixels are present.
    prob_high (float, optional, default=99.9): The probability of exceedance used to calculate the upper percentile for intensity rescaling. The default value is 99.9, meaning that the highest 0.1% of intensities will be clipped.

    Returns:
    im_adj (numpy.array): The rescaled image with increased intensity for the selected bands. The dimensions and number of bands of the output image may be different from the input image.
    """
    return SDS_preprocess.rescale_image_intensity(
        im_ms[:, :, [2, 1, 0]], cloud_mask, prob_high
    )


def create_color_mapping_as_ints(int_list: list[int]) -> dict:
    """
    This function creates a color mapping dictionary for a given list of integers, assigning a unique RGB color to each integer. The colors are generated using the HLS color model, and the resulting RGB values are integers in the range of 0-255.

    Arguments:

    int_list (list): A list of integers for which unique colors need to be generated.
    Returns:

    color_mapping (dict): A dictionary where the keys are the input integers and the values are the corresponding RGB colors as tuples of integers.
    """
    n = len(int_list)
    h_step = 1.0 / n
    color_mapping = {}

    for i, num in enumerate(int_list):
        h = i * h_step
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(h, 0.5, 1.0)]
        color_mapping[num] = (r, g, b)

    return color_mapping


def create_color_mapping(int_list: list[int]) -> dict:
    """
    This function creates a color mapping dictionary for a given list of integers, assigning a unique RGB color to each integer. The colors are generated using the HLS color model, and the resulting RGB values are floating-point numbers in the range of 0.0-1.0.

    Arguments:

    int_list (list): A list of integers for which unique colors need to be generated.
    Returns:

    color_mapping (dict): A dictionary where the keys are the input integers and the values are the corresponding RGB colors as tuples of floating-point numbers.
    """
    n = len(int_list)
    h_step = 1.0 / n
    color_mapping = {}

    for i, num in enumerate(int_list):
        h = i * h_step
        r, g, b = [
            x for x in colorsys.hls_to_rgb(h, 0.5, 1.0)
        ]  # Removed the int() conversion and * 255
        color_mapping[num] = (r, g, b)

    return color_mapping


def create_classes_overlay_image(labels):
    """
    Creates an overlay image by mapping class labels to colors.

    Args:
    labels (numpy.ndarray): A 2D array representing class labels for each pixel in an image.

    Returns:
    numpy.ndarray: A 3D array representing an overlay image with the same size as the input labels.
    """
    # Ensure that the input labels is a NumPy array
    labels = np.asarray(labels)

    # Make an overlay the same size of the image with 3 color channels
    overlay_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.float32)

    # Create a color mapping for the labels
    class_indices = np.unique(labels)
    color_mapping = create_color_mapping(class_indices)

    # Create the overlay image by assigning the color for each label
    for index, class_color in color_mapping.items():
        overlay_image[labels == index] = class_color

    return overlay_image


def plot_image_with_legend(
    original_image: "np.ndarray[float]",
    merged_overlay: "np.ndarray[float]",
    all_overlay: "np.ndarray[float]",
    pixelated_shoreline: "np.ndarray[float]",
    merged_legend: list,
    all_legend: list,
    im_ref_buffer: np.ndarray[float],
    titles: list[str] = [],
    pixelated_shoreline_extraction_area: np.ndarray[float] = None,
):
    """
    Plots the original image, merged classes, and all classes with their corresponding legends.

    Args:
    original_image (numpy.ndarray): The original image. Must be a 2D or 3D numpy array.
    merged_overlay (numpy.ndarray): The image with merged classes overlay. Must be a numpy array with the same shape as original_image.
    all_overlay (numpy.ndarray): The image with all classes overlay. Must be a numpy array with the same shape as original_image.
    pixelated_shoreline (numpy.ndarray): The pixelated shoreline points. Must be a 2D numpy array where each row represents a point.
    merged_legend (list): A list of legend handles for the merged classes. Each handle must be a matplotlib artist.
    all_legend (list): A list of legend handles for all classes. Each handle must be a matplotlib artist.
    titles (list, optional): A list of titles for the subplots. Must contain three strings if provided. Defaults to ["Original Image", "Merged Classes", "All Classes"].
    im_ref_buffer (numpy.ndarray): A 2D numpy array with the same shape as original_image. The array should have True values where reference shoreline pixels are present.

    Returns:
    matplotlib.figure.Figure: The resulting figure.
    """
    
    if not titles or len(titles) != 3:
        titles = ["Original Image", "Merged Classes", "All Classes"]
    fig = plt.figure()
    fig.set_size_inches([18, 9])

    if original_image.shape[1] > 2.5 * original_image.shape[0]:
        gs = gridspec.GridSpec(3, 1)
    else:
        gs = gridspec.GridSpec(1, 3)

    # Create a masked array where False values are masked
    masked_array = None
    if im_ref_buffer is not None:
        masked_array = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
    # color map for the reference shoreline buffer
    masked_cmap = plt.get_cmap("PiYG")

    # if original_image is wider than 2.5 times as tall, plot the images in a 3x1 grid (vertical)
    if original_image.shape[0] > 2.5 * original_image.shape[1]:
        # vertical layout 3x1
        gs = gridspec.GridSpec(3, 1)
        ax2_idx, ax3_idx = (1, 0), (2, 0)
        bbox_to_anchor = (1.05, 0.5)
        loc = "center left"
    else:
        # horizontal layout 1x3
        gs = gridspec.GridSpec(1, 3)
        ax2_idx, ax3_idx = (0, 1), (0, 2)
        bbox_to_anchor = (0.5, -0.23)
        loc = "lower center"

    gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[ax2_idx], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[ax3_idx], sharex=ax1, sharey=ax1)

    # Plot original image
    ax1.imshow(original_image)
    ax1.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
    for idx in range(len(pixelated_shoreline_extraction_area)):
        ax1.plot(pixelated_shoreline_extraction_area[idx][:, 0], pixelated_shoreline_extraction_area[idx][:, 1], color='#cb42f5', markersize=1)
    ax1.set_title(titles[0])
    ax1.axis("off")

    # Plot the second image that has the merged the water classes and all the land classes together
    ax2.imshow(merged_overlay)
    # Plot the reference shoreline buffer
    if masked_array is not None:
        ax2.imshow(masked_array, cmap=masked_cmap, alpha=0.60)
    ax2.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
    for idx in range(len(pixelated_shoreline_extraction_area)):
        ax2.plot(pixelated_shoreline_extraction_area[idx][:, 0], pixelated_shoreline_extraction_area[idx][:, 1], color='#cb42f5', markersize=1)
    ax2.set_title(titles[1])
    ax2.axis("off")
    if merged_legend:  # Check if the list is not empty
        ax2.legend(
            handles=merged_legend,
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            borderaxespad=0.0,
        )

    # Plot the second image that shows all the classes separately
    ax3.imshow(all_overlay)
    if masked_array is not None:
        ax3.imshow(masked_array, cmap=masked_cmap, alpha=0.60)
    ax3.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
    for idx in range(len(pixelated_shoreline_extraction_area)):
        ax3.plot(pixelated_shoreline_extraction_area[idx][:, 0], pixelated_shoreline_extraction_area[idx][:, 1], color='#cb42f5', markersize=1)
    ax3.set_title(titles[2])
    ax3.axis("off")
    if all_legend:  # Check if the list is not empty
        ax3.legend(
            handles=all_legend,
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            borderaxespad=0.0,
        )

    # Return the figure object
    return fig


def save_detection_figure(fig, filepath: str, date: str, satname: str) -> None:
    """
    Save the given figure as a jpg file with a specified dpi.

    Args:
    fig (Figure): The figure object to save.
    filepath (str): The directory path where the image will be saved.
    date (str): The date the satellite image was taken in the format 'YYYYMMDD'.
    satname (str): The name of the satellite that took the image.

    Returns:
    None
    """
    fig.savefig(
        os.path.join(filepath, date + "_" + satname + ".jpg"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure after saving
    plt.close("all")
    del fig


def create_legend(
    class_mapping: dict, color_mapping: dict = None, additional_patches: list = None
) -> list[mpatches.Patch]:
    """
    Creates a list of legend patches using class and color mappings.

    Args:
    class_mapping (dict): A dictionary mapping class indices to class names.
    color_mapping (dict, optional): A dictionary mapping class indices to colors. Defaults to None.
    additional_patches (list, optional): A list of additional patches to be appended to the legend. Defaults to None.

    Returns:
    list: A list of legend patches.
    """
    if color_mapping is None:
        color_mapping = create_color_mapping_as_ints(class_mapping.keys())

    legend = [
        mpatches.Patch(
            color=np.array(color) / 255, label=f"{class_mapping.get(index, f'{index}')}"
        )
        for index, color in color_mapping.items()
    ]

    return legend + additional_patches if additional_patches else legend


def create_overlay(
    im_RGB: "np.ndarray[float]",
    im_labels: "np.ndarray[int]",
    overlay_opacity: float = 0.35,
) -> "np.ndarray[float]":
    """
    Create an overlay on the given image using the provided labels and
    specified overlay opacity.

    Args:
    im_RGB (np.ndarray[float]): The input image as an RGB numpy array (height, width, 3).
    im_labels (np.ndarray[int]): The array containing integer labels of the same dimensions as the input image.
    overlay_opacity (float, optional): The opacity value for the overlay (default: 0.35).

    Returns:
    np.ndarray[float]: The combined numpy array of the input image and the overlay.
    """
    # Create an overlay using the given labels
    overlay = create_classes_overlay_image(im_labels)
    # Combine the original image and the overlay using the correct opacity
    combined_float = im_RGB * (1 - overlay_opacity) + overlay * overlay_opacity
    return combined_float


def shoreline_detection_figures(
    im_ms: np.ndarray,
    cloud_mask: "np.ndarray[bool]",
    merged_labels: np.ndarray,
    all_labels: np.ndarray,
    shoreline: np.ndarray,
    image_epsg: str,
    georef,
    settings: dict,
    date: str,
    satname: str,
    class_mapping: dict,
    save_location: str = "",
    im_ref_buffer: np.ndarray = None,
    shoreline_extraction_area:np.ndarray=None,
):
    """
    Creates shoreline detection figures with overlays and saves them as JPEG files.

    Args:
    im_ms (numpy.ndarray): The multispectral image.
    cloud_mask (numpy.ndarray): The cloud mask.
    merged_labels (numpy.ndarray): The merged class labels.
    all_labels (numpy.ndarray): All class labels.
    shoreline (numpy.ndarray): The shoreline points.
    image_epsg (str): The EPSG code of the image.
    georef (numpy.ndarray): The georeference matrix.
    settings (dict): The settings dictionary.
    date (str): The date of the image.
    satname (str): The satellite name.
    class_mapping (dict): A dictionary mapping class indices to class names.
    save_location (str, optional): The directory path where the images will be saved. Defaults to "".
    im_ref_buffer (numpy.ndarray, optional): The reference shoreline buffer. Defaults to None.
    shoreline_extraction_area (numpy.ndarray, optional): The area where the shoreline was extracted. Defaults to None.
    """
    sitename = settings["inputs"]["sitename"]
    if save_location:
        filepath = os.path.join(save_location, "jpg_files", "detection")
    else:
        filepath_data = settings["inputs"]["filepath"]
        filepath = os.path.join(filepath_data, sitename, "jpg_files", "detection")
        
        
    os.makedirs(filepath, exist_ok=True)
    # logger.info(f"shoreline_detection_figures filepath: {filepath}")
    logger.info(f"im_ref_buffer.shape: {im_ref_buffer.shape}")

    # increase the intensity of the image for visualization
    im_RGB = increase_image_intensity(im_ms, cloud_mask, prob_high=99.9)


    im_merged = create_overlay(im_RGB, merged_labels, overlay_opacity=0.35)
    im_all = create_overlay(im_RGB, all_labels, overlay_opacity=0.35)


    # Mask clouds in the images
    im_RGB, im_merged, im_all = mask_clouds_in_images(
        im_RGB, im_merged, im_all, cloud_mask
    )

    # Convert shoreline points to pixel coordinates
    try:
        pixelated_shoreline = SDS_tools.convert_world2pix(
            SDS_tools.convert_epsg(shoreline, settings["output_epsg"], image_epsg)[
                :, [0, 1]
            ],
            georef,
        )
    except:
        pixelated_shoreline = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    # Convert shoreline extraction area to pixel coordinates
    shoreline_extraction_area_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    shoreline_extraction_area_pix  = []
    if shoreline_extraction_area is not None:
        if len(shoreline_extraction_area) == 0:
            shoreline_extraction_area = None
    
    if shoreline_extraction_area is not None:
        shoreline_extraction_area_pix  = []
        for idx in range(len(shoreline_extraction_area)):
            shoreline_extraction_area_pix.append(
                SDS_preprocess.transform_world_coords_to_pixel_coords(shoreline_extraction_area[idx],settings["output_epsg"], georef, image_epsg)
            )
    # Create legend for the shorelines
    black_line = mlines.Line2D([], [], color="k", linestyle="-", label="shoreline")
    buffer_patch = mpatches.Patch(
        color="#800000", alpha=0.80, label="Reference shoreline buffer"
    )
    # The additional patches to be appended to the legend
    additional_legend_items = [black_line, buffer_patch]
    
    if shoreline_extraction_area is not None:
        shoreline_extraction_area_line = mlines.Line2D([], [], color="#cb42f5", linestyle="-", label="shoreline extraction area")
        additional_legend_items.append(shoreline_extraction_area_line)

    # create a legend for the class colors and the shoreline
    all_classes_legend = create_legend(
        class_mapping, additional_patches=additional_legend_items
    )
    merged_classes_legend = create_legend(
        class_mapping={0: "other", 1: "water"},
        additional_patches=additional_legend_items,
    )

    # Plot images
    fig = plot_image_with_legend(
        im_RGB,
        im_merged,
        im_all,
        pixelated_shoreline,
        merged_classes_legend,
        all_classes_legend,
        im_ref_buffer,
        titles=[sitename, date, satname],
        pixelated_shoreline_extraction_area=shoreline_extraction_area_pix,
    )
    # save a .jpg under /jpg_files/detection
    save_detection_figure(fig, filepath, date, satname)
    plt.close(fig)


def mask_clouds_in_images(
    im_RGB: "np.ndarray[float]",
    im_merged: "np.ndarray[float]",
    im_all: "np.ndarray[float]",
    cloud_mask: "np.ndarray[bool]",
):
    """
    Applies a cloud mask to three input images (im_RGB, im_merged & im_all) by setting the
    cloudy portions to a value of 1.0.

    Args:
        im_RGB (np.ndarray[float]): An RGB image, with shape (height, width, 3).
        im_merged (np.ndarray[float]): A merged image, with the same shape as im_RGB.
        im_all (np.ndarray[float]): An 'all' image, with the same shape as im_RGB.
        cloud_mask (np.ndarray[bool]): A boolean cloud mask, with shape (height, width).

    Returns:
        tuple: A tuple containing the masked im_RGB, im_merged and im_all images.
    """
    nan_color_float = 1.0
    new_cloud_mask = np.repeat(cloud_mask[:, :, np.newaxis], im_RGB.shape[2], axis=2)

    im_RGB[new_cloud_mask] = nan_color_float
    im_merged[new_cloud_mask] = nan_color_float
    im_all[new_cloud_mask] = nan_color_float

    return im_RGB, im_merged, im_all


def simplified_find_contours(
    im_labels: np.array, cloud_mask: np.array, reference_shoreline_buffer: np.array
) -> List[np.array]:
    """Find contours in a binary image using skimage.measure.find_contours and processes out contours that contain NaNs.
    Parameters:
    -----------
    im_labels: np.nd.array
        binary image with 0s and 1s
    cloud_mask: np.array
        boolean array indicating cloud mask
    Returns:
    -----------
    processed_contours: list of arrays
        processed image contours (only the ones that do not contains NaNs)
    """
    # make a copy of the im_labels array as a float (this allows find contours to work))
    im_labels_masked = im_labels.copy().astype(float)
    # Apply the cloud mask by setting masked pixels to NaN
    im_labels_masked[cloud_mask] = np.NaN
    # only keep the pixels inside the reference shoreline buffer
    im_labels_masked[~reference_shoreline_buffer] = np.NaN
    
    # 0 or 1 labels means 0.5 is the threshold
    contours = measure.find_contours(im_labels_masked, 0.5)

    # remove contour points that are NaNs (around clouds and nodata intersections)
    processed_contours = SDS_shoreline.process_contours(contours)

    return processed_contours


def convert_date_column_to_datetime(
    gdf: gpd.GeoDataFrame, date_column: str, timezone: str = 'UTC'
) -> gpd.GeoDataFrame:
    """
    Converts the date column of a GeoDataFrame to datetime format with timezone information and converts to datetime.datetime in UTC.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the date column.
        date_column (str): The name of the date column to convert.
        timezone (str): The timezone to which naive datetime entries should be localized. Default is 'UTC'.

    Returns:
        gpd.GeoDataFrame: The updated GeoDataFrame with the date column in datetime format with timezone.
    """
    # Ensure the date column is in datetime format
    gdf[date_column] = pd.to_datetime(gdf[date_column], errors="coerce")

    # Drop any rows where the date is NaT
    gdf = gdf.dropna(subset=[date_column])

    # Convert the date column to string format
    gdf[date_column] = gdf[date_column].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Convert the date column back to datetime in UTC
    gdf[date_column] = pd.to_datetime(gdf[date_column], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.tz_localize('UTC')

    return gdf



@time_func
def extract_shorelines_with_dask(
    session_path: str,
    metadata: dict,
    settings: dict,
    class_indices: list = None,
    class_mapping: dict = None,
    save_location: str = "",
    shoreline_extraction_area: gpd.GeoDataFrame = None,
    **kwargs: dict,
) -> dict:
    """
    Extracts shorelines from satellite imagery using a Dask-based implementation.

    Args:
        session_path (str): The path to the session directory.
        metadata (dict): A dictionary containing metadata for the satellite imagery.
        settings (dict): A dictionary containing settings for the shoreline extraction.
        class_indices (list, optional): A list of class indices to extract. Defaults to None.
        class_mapping (dict, optional): A dictionary mapping class indices to class names. Defaults to None.
        save_location (str, optional): The path to save the extracted shorelines. Defaults to "".
        shoreline_extraction_area (gpd.GeoDataFrame, optional): A GeoDataFrame containing the area where the shoreline was extracted. Defaults to None.
        **kwargs (dict): Additional keyword arguments.

    Returns:
        dict: A dictionary containing the extracted shorelines for each satellite.
    """
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]

    # create a subfolder to store the .jpg images showing the detection
    if not save_location:
        filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
        os.makedirs(filepath_jpg, exist_ok=True)

    # get the directory containing the good model outputs
    good_folder = get_sorted_model_outputs_directory(session_path)

    # get the list of files that were sorted as 'good'
    filtered_files = get_filtered_files_dict(good_folder, "npz", sitename)
    # keep only the metadata for the files that were sorted as 'good'
    metadata = edit_metadata(metadata, filtered_files)

    for satname in metadata.keys():
        if not metadata[satname]:
            logger.warning(f"metadata['{satname}'] is empty")
        else:
            logger.info(
                f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('epsg',[]))} of epsg: {np.unique(metadata[satname].get('epsg',[]))}"
            )
            logger.info(
                f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('dates',[]))} of dates Sample first five: {list(islice(metadata[satname].get('dates',[]),5))}"
            )
            logger.info(
                f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('filenames',[]))} of filenames Sample first five: {list(islice(metadata[satname].get('filenames',[]),5))}"
            )
            logger.info(
                f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('im_dimensions',[]))} of im_dimensions: {np.unique(metadata[satname].get('im_dimensions',[]))}"
            )
            logger.info(
                f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('acc_georef',[]))} of acc_georef: {np.unique(metadata[satname].get('acc_georef',[]))}"
            )
            logger.info(
                f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('im_quality',[]))} of im_quality: {np.unique(metadata[satname].get('im_quality',[]))}"
            )

    shoreline_dict = {
        "dates": [],
        "shorelines": [],
        "cloud_cover": [],
        "geoaccuracy": [],
        "idx": [],
        "filename": [],
        "satname": [],
    }
    all_satellite_gdfs = []
    for satname in metadata.keys():
        satellite_gdf = process_satellite(
            satname,
            settings,
            metadata,
            session_path,
            class_indices,
            class_mapping,
            save_location,
            batch_size=10,
            shoreline_extraction_area=shoreline_extraction_area,
            **kwargs,
        )
        if satellite_gdf is not None:
            all_satellite_gdfs.append(satellite_gdf)

    # combine the extracted shorelines for each satellite
    all_shorelines_gdf = concat_and_sort_geodataframes(all_satellite_gdfs, "date", "UTC")
    if all_shorelines_gdf.empty:
        print("No shorelines were extracted.")
        logger.warning("No shorelines were extracted.")
        return {}
    all_shorelines_gdf = all_shorelines_gdf.reset_index(drop=True)
    # convert to epsg 4326
    all_shorelines_gdf_4326 = all_shorelines_gdf.to_crs(epsg=4326)
    # drop the filename column
    all_shorelines_gdf_4326.drop(columns=["filename"],inplace=True)
    # print(f"all_shorelines_gdf_4326: {all_shorelines_gdf_4326}")

    # Save extracted shorelines to GeoJSON files
    all_shorelines_gdf_4326.to_file(
        os.path.join(session_path, 'extracted_shorelines_lines.geojson'), driver="GeoJSON"
    )

    print(f"extracted_shorelines_lines.geojson saved to {os.path.join(session_path, 'extracted_shorelines_lines.geojson')}")

    # convert linestrings to multipoints
    points_gdf = convert_linestrings_to_multipoints(all_shorelines_gdf_4326)
    projected_gdf = stringify_datetime_columns(points_gdf)
    # Save extracted shorelines as a GeoJSON file
    projected_gdf.to_file(
        os.path.join(session_path, 'extracted_shorelines_points.geojson'), driver="GeoJSON"
    )

    print(f"extracted_shorelines_points.geojson saved to {os.path.join(session_path, 'extracted_shorelines_points.geojson')}")

    # convert the extracted shorelines dates to ISO 8601 format
    all_shorelines_gdf = convert_date_column_to_datetime(all_shorelines_gdf, 'date')
    # create a dictionary of the extracted shorelines
    for date, group in all_shorelines_gdf.groupby("date"):
        shorelines = [np.array(geom.coords) for geom in group.geometry]
        contours_array = extract_contours(shorelines)
        shoreline_dict["shorelines"].append(contours_array)
        shoreline_dict["dates"].append(date.to_pydatetime())
        # Append values for each group, ensuring they are correctly extracted
        shoreline_dict["cloud_cover"].append(group["cloud_cover"].values[0])
        shoreline_dict["geoaccuracy"].append(group["geoaccuracy"].values[0])
        shoreline_dict["idx"].append(group["idx"].values[0])
        shoreline_dict["filename"].append(group["filename"].values[0])
        shoreline_dict["satname"].append(group["satname"].values[0])

    return shoreline_dict


def get_sorted_model_outputs_directory(
    session_path: str,
) -> str:
    """
    Sort model output files into "good" and "bad" folders based on the satellite name in the filename.
    Applies the land mask to the model output files in the "good" folder.

    Args:
        session_path (str): The path to the session directory containing the model output files.

    Returns:
        str: The path to the "good" folder containing the sorted model output files.
    """
    # for each satellite, sort the model outputs into good & bad
    good_folder = os.path.join(session_path, "good")
    bad_folder = os.path.join(session_path, "bad")
    # empty the good and bad folders 
    # if os.path.exists(good_folder):
    #     shutil.rmtree(good_folder)
    # if os.path.exists(bad_folder):
    #     shutil.rmtree(bad_folder)
        
    os.makedirs(good_folder, exist_ok=True)  # Ensure good_folder exists.
    os.makedirs(bad_folder, exist_ok=True)   # Ensure bad_folder exists.
    
    satellites = get_satellites_in_directory(session_path)
    print(f"Satellites in directory: {satellites}")
    for satname in satellites:
        print(f"Filtering model outputs for {satname}")
        # Define the pattern for matching files related to the current satellite.
        pattern = f".*{re.escape(satname)}.*\\.npz$"  # Match files with the satellite name in the filename.
        # search the session path for the satellite files
        search_path = session_path
        files = []
        try:
            # Retrieve the list of relevant .npz files.
            files = file_utilities.find_files_in_directory(search_path, pattern, raise_error=False)
            logger.info(f"{search_path} contains {len(files)} files for satellite {satname}")
        except Exception as e:
            logger.error(f"Error finding files for satellite {satname}: {e}")
            continue  # Skip to the next satellite if there's an issue.

        logger.info(f"{session_path} contained {satname} files: {len(files)} ")
        
        # If there are files sort the files into good and bad folders
        filter_model_outputs(satname, files, good_folder, bad_folder)
        # Apply the land mask if there are files in the good folder.
        # if os.listdir(good_folder):
        #     apply_land_mask(good_folder)
            
    return good_folder


def get_min_shoreline_length(satname: str, default_min_length_sl: float) -> int:
    """
    Given a satellite name and a default minimum shoreline length, returns the minimum shoreline length
    for the specified satellite.

    If the satellite name is "L7", the function returns a minimum shoreline length of 200, as this
    satellite has diagonal bands that require a shorter minimum length. For all other satellite names,
    the function returns the default minimum shoreline length.

    Args:
    - satname (str): A string representing the name of the satellite to retrieve the minimum shoreline length for.
    - default_min_length_sl (float): A float representing the default minimum shoreline length to be returned if
                                      the satellite name is not "L7".

    Returns:
    - An integer representing the minimum shoreline length for the specified satellite.

    Example usage:
    >>> get_min_shoreline_length("L5", 500)
    500
    >>> get_min_shoreline_length("L7", 500)
    200
    """
    # reduce min shoreline length for L7 because of the diagonal bands
    if satname == "L7":
        return 200
    else:
        return default_min_length_sl


def get_pixel_size_for_satellite(satname: str) -> int:
    """Returns the pixel size of a given satellite.
    ["L5", "L7", "L8", "L9"] = 15 meters
    "S2" = 10 meters

    Args:
        satname (str): A string indicating the name of the satellite.

    Returns:
        int: The pixel size of the satellite in meters.

    Raises:
        None.
    """
    if satname in ["L5", "L7", "L8", "L9"]:
        pixel_size = 15
    elif satname == "S2":
        pixel_size = 10
    return pixel_size


def load_extracted_shoreline_from_files(
    dir_path: str,
) -> Optional["Extracted_Shoreline"]:
    """
    Load the extracted shoreline from the given directory.

    The function searches the directory for the extracted shoreline GeoJSON file, the shoreline settings JSON file,
    and the extracted shoreline dictionary JSON file. If any of these files are missing, the function returns None.

    Args:
        dir_path: The path to the directory containing the extracted shoreline files.

    Returns:
        An instance of the Extracted_Shoreline class containing the extracted shoreline data, or None if any of the
        required files are missing.
    """
    required_files = {
        "geojson": "*shoreline*.geojson",
        "settings": "*shoreline*settings*.json",
        "dict": "*shoreline*dict*.json",
    }

    extracted_files = {}
    logger.info(f"Loading extracted shorelines from: {dir_path}")
    for file_type, file_pattern in required_files.items():
        file_paths = glob(os.path.join(dir_path, file_pattern))
        if not file_paths:
            logger.warning(f"No {file_type} file could be loaded from {dir_path}")
            return None

        file_path = file_paths[0]  # Use the first file if there are multiple matches
        if file_type == "geojson":
            extracted_files[file_type] = geodata_processing.read_gpd_file(file_path)
        else:
            extracted_files[file_type] = file_utilities.load_data_from_json(file_path)

    extracted_shorelines = Extracted_Shoreline()
    # attempt to load the extracted shorelines from the files. If there is an error, return None
    try:
        extracted_shorelines = extracted_shorelines.load_extracted_shorelines(
            extracted_files["dict"],
            extracted_files["settings"],
            extracted_files["geojson"],
        )
    except ValueError as e:
        logger.error(f"Error loading extracted shorelines: {e}")
        del extracted_shorelines
        return None

    return extracted_shorelines


class Extracted_Shoreline:
    """Extracted_Shoreline: contains the extracted shorelines within a Region of Interest (ROI)"""

    LAYER_NAME = "extracted_shoreline"
    FILE_NAME = "extracted_shorelines.geojson"

    def __init__(
        self,
    ):
        # gdf: geodataframe containing extracted shoreline for ROI_id
        self.gdf = gpd.GeoDataFrame()
        # Use roi id to identify which ROI extracted shorelines derive from
        self.roi_id = ""
        # dictionary : dictionary of extracted shorelines
        # contains keys 'dates', 'shorelines', 'filename', 'cloud_cover', 'geoaccuracy', 'idx', 'MNDWI_threshold', 'satname'
        self.dictionary = {}
        # shoreline_settings: dictionary of settings used to extract shoreline
        self.shoreline_settings = {}

    def __str__(self):
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 3 rows as a string
        first_rows = self.gdf
        geom_str = ""
        if isinstance(self.gdf, gpd.GeoDataFrame):
            if "geometry" in self.gdf.columns:
                first_rows = self.gdf.head(3).drop(columns="geometry").to_string()
            if not self.gdf.empty:
                geom_str = str(self.gdf.iloc[0]["geometry"])[:100] + "...)"
        # Get CRS information
        crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
        return f"Extracted Shoreline:\nROI ID: {self.roi_id}\ngdf:\n{crs_info}\nColumns and Data Types:\n{col_info}\n\nFirst 3 Rows:\n{first_rows}\n geometry: {geom_str}"

    def __repr__(self):
        # Get column names and their data types
        col_info = self.gdf.dtypes.apply(lambda x: x.name).to_string()
        # Get first 5 rows as a string
        first_rows = self.gdf
        geom_str = ""
        if isinstance(self.gdf, gpd.GeoDataFrame):
            if "geometry" in self.gdf.columns:
                first_rows = self.gdf.head(3).drop(columns="geometry").to_string()
            if not self.gdf.empty:
                geom_str = str(self.gdf.iloc[0]["geometry"])[:100] + "...)"
        # Get CRS information
        crs_info = f"CRS: {self.gdf.crs}" if self.gdf.crs else "CRS: None"
        return f"Extracted Shoreline:\nROI ID: {self.roi_id}\ngdf:\n{crs_info}\nColumns and Data Types:\n{col_info}\n\nFirst 3 Rows:\n{first_rows}\n geometry: {geom_str}"

    def get_roi_id(self) -> Optional[str]:
        """
        Extracts the region of interest (ROI) ID from the shoreline settings.

        The method retrieves the sitename field from the shoreline settings inputs dictionary and extracts the
        ROI ID from it, if present. The sitename field is expected to be in the format "ID_XXXX_datetime03-22-23__07_29_15",
        where XXXX is the id of the ROI. If the sitename field is not present or is not in the
        expected format, the method returns None.

        shoreline_settings:
        {
            'inputs' {
                "sitename": 'ID_0_datetime03-22-23__07_29_15',
                }
        }

        Returns:
            The ROI ID as a string, or None if the sitename field is not present or is not in the expected format.
        """
        inputs = self.shoreline_settings.get("inputs", {})
        sitename = inputs.get("sitename", "")
        # checks if the ROI ID is present in the 'sitename' saved in the shoreline settings
        roi_id = sitename.split("_")[1] if sitename else None
        return roi_id

    def remove_selected_shorelines(
        self, dates: list[datetime.datetime], satellites: list[str]
    ) -> None:
        """
        Removes selected shorelines based on the provided dates and satellites.

        Args:
            dates (list[datetime.datetime]): A list of dates to filter the shorelines.
            satellites (list[str]): A list of satellites to filter the shorelines.

        Returns:
            None
        """
        if hasattr(self, "dictionary"):
            self._remove_from_dict(dates, satellites)
        if hasattr(self, "gdf"):
            if not self.gdf.empty:
                self.gdf = self._remove_from_gdf(dates, satellites)

    def _remove_from_dict(
        self, dates: list["datetime.datetime"], satellites: list[str]
    ) -> dict:
        """
        Remove selected indexes from the dictionary based on the dates and satellites passed in for a specific region of interest.

        Args:
            dates (list['datetime.datetime']): The list of dates to filter.
            satellites (list[str]): The list of satellites to filter.

        Returns:
            dict: The updated dictionary for the specified region of interest.
        """
        selected_indexes = common.get_selected_indexes(
            self.dictionary, dates, satellites
        )
        self.dictionary = common.delete_selected_indexes(
            self.dictionary, selected_indexes
        )
        return self.dictionary

    def _remove_from_gdf(
        self, dates: list["datetime.datetime"], satellites: list[str]
    ) -> gpd.GeoDataFrame:
        """
        Remove rows from the GeoDataFrame based on the specified dates and satellites.

        Args:
            dates (list[datetime.datetime]): A list of datetime objects representing the dates to filter.
            satellites (list[str]): A list of satellite names to filter.

        Returns:
            gpd.GeoDataFrame: The updated GeoDataFrame after removing the matching rows.
        """
        if all(isinstance(date, datetime.date) for date in dates):
            dates = [date.strftime("%Y-%m-%d %H:%M:%S") for date in dates]

        for sat, date in zip(satellites, dates):
            matching_rows = self.gdf[
                (self.gdf["satname"] == sat) & (self.gdf["date"] == date)
            ]
            self.gdf = self.gdf.drop(matching_rows.index)
        return self.gdf

    def load_extracted_shorelines(
        self,
        extracted_shoreline_dict: dict = None,
        shoreline_settings: dict = None,
        extracted_shorelines_gdf: gpd.GeoDataFrame = None,
    ):
        """Loads extracted shorelines into the Extracted_Shoreline class.
        Intializes the class with the extracted shorelines dictionary, shoreline settings, and the extracted shorelines geodataframe

        Args:
            extracted_shoreline_dict (dict, optional): A dictionary containing the extracted shorelines. Defaults to None.
            shoreline_settings (dict, optional): A dictionary containing the shoreline settings. Defaults to None.
            extracted_shorelines_gdf (GeoDataFrame, optional): The extracted shorelines in a GeoDataFrame. Defaults to None.

        Returns:
            object: The Extracted_Shoreline class with the extracted shorelines loaded.

        Raises:
            ValueError: If the input arguments are invalid.
        """

        if not isinstance(extracted_shoreline_dict, dict):
            raise ValueError(
                f"extracted_shoreline_dict must be dict. not {type(extracted_shoreline_dict)}"
            )
        if extracted_shoreline_dict == {}:
            raise ValueError("extracted_shoreline_dict cannot be empty.")

        if extracted_shorelines_gdf is not None:
            if not isinstance(extracted_shorelines_gdf, gpd.GeoDataFrame):
                raise ValueError(
                    f"extracted_shorelines_gdf must be valid geodataframe. not {type(extracted_shorelines_gdf)}"
                )
            if extracted_shorelines_gdf.empty:
                raise ValueError("extracted_shorelines_gdf cannot be empty.")
            self.gdf = extracted_shorelines_gdf

        if not isinstance(shoreline_settings, dict):
            raise ValueError(
                f"shoreline_settings must be dict. not {type(shoreline_settings)}"
            )
        if shoreline_settings == {}:
            raise ValueError("shoreline_settings cannot be empty.")

        # dictionary : dictionary of extracted shorelines
        self.dictionary = extracted_shoreline_dict
        # shoreline_settings: dictionary of settings used to extract shoreline
        self.shoreline_settings = shoreline_settings
        # Use roi id to identify which ROI extracted shorelines derive from
        self.roi_id = shoreline_settings["inputs"]["roi_id"]
        return self

    def create_extracted_shorelines(
        self,
        roi_id: str = None,
        shoreline: gpd.GeoDataFrame = None,
        roi_settings: dict = None,
        settings: dict = None,
        output_directory:str = None,
        shoreline_extraction_area: gpd.GeoDataFrame = None,
    ) -> "Extracted_Shoreline":
        """
        Extracts shorelines for a specified region of interest (ROI) and returns an Extracted_Shoreline class instance.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): A dictionary of region of interest settings.
        - settings (dict): A dictionary of extraction settings.
        - output_directory (str): The path to the directory where the extracted shorelines will be saved.
           - detection figures will be saved in a subfolder called 'jpg_files' within the output_directory.
           - extract_shoreline reports will be saved within the output_directory.

        Returns:
        - object: The Extracted_Shoreline class instance.
        """
        # validate input parameters are not empty and are of the correct type
        self._validate_input_params(roi_id, shoreline, roi_settings, settings)

        logger.info(f"Extracting shorelines for ROI id{roi_id}")
        # extract the shorelines using the settings doing so returns a dictionary with all the shorelines
        self.dictionary = self.extract_shorelines(
            shoreline,
            roi_settings,
            settings,
            output_directory=output_directory,
            shoreline_extraction_area = shoreline_extraction_area
        )
        if self.dictionary == {}:
            logger.warning(f"No extracted shorelines for ROI {roi_id}")
            raise exceptions.No_Extracted_Shoreline(roi_id)

        if is_list_empty(self.dictionary["shorelines"]):
            logger.warning(f"No extracted shorelines for ROI {roi_id}")
            raise exceptions.No_Extracted_Shoreline(roi_id)

        self.gdf = self.create_geodataframe(
            self.shoreline_settings["output_epsg"],
            output_crs="EPSG:4326",
            geomtype="points",
        )
        return self

    def create_extracted_shorelines_from_session(
        self,
        roi_id: str = None,
        shoreline: gpd.GeoDataFrame = None,
        roi_settings: dict = None,
        settings: dict = None,
        session_path: str = None,
        new_session_path: str = None,
        output_directory: str = None, 
        shoreline_extraction_area : gpd.geodataframe = None,  
        **kwargs: dict,
    ) -> "Extracted_Shoreline":
        """
        Extracts shorelines for a specified region of interest (ROI) from a saved session and returns an Extracted_Shoreline class instance.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): Dictionary containing settings for the ROI. It must have the following keys:
            {
                "dates": ["2018-12-01", "2019-03-01"],
                "sat_list": ["L8", "L9", "S2"],
                "roi_id": "lyw1",
                "polygon": [
                [
                    [-73.94584118213996, 40.57245559853209],
                    [-73.94584118213996, 40.52844804565595],
                    [-73.87282173497694, 40.52844804565595],
                    [-73.87282173497694, 40.57245559853209],
                    [-73.94584118213996, 40.57245559853209]
                ]
                ],
                "landsat_collection": "C02",
                "sitename": "ID_lyw1_datetime01-18-24__12_26_51",
                "filepath": "C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
            },
        - settings (dict): A dictionary of extraction settings.
        - session_path (str): The path of the saved session from which the shoreline extraction needs to be resumed.
        - new_session_path (str) :The path of the new session where the extreacted shorelines extraction will be saved
        - output_directory (str): The path to the directory where the extracted shorelines will be saved.
            - detection figures will be saved in a subfolder called 'jpg_files' within the output_directory.
            - extract_shoreline reports will be saved within the output_directory.
        - shoreline_extraction_area (gpd.geodataframe, optional): A GeoDataFrame containing the area to extract shorelines from. Defaults to None.
        Returns:
        - object: The Extracted_Shoreline class instance.
        """
        # validate input parameters are not empty and are of the correct type
        self._validate_input_params(roi_id, shoreline, roi_settings, settings)

        logger.info(f"Extracting shorelines for ROI id: {roi_id}")

        # read model settings from session path
        model_settings_path = os.path.join(session_path, "model_settings.json")
        model_settings = file_utilities.read_json_file(
            model_settings_path, raise_error=True
        )
        # get model type from model settings
        model_type = model_settings.get("model_type", "")
        if model_type == "":
            raise ValueError(
                f"Model type cannot be empty.{model_settings_path} did not contain model_type key."
            )
        # read model card from downloaded models path
        downloaded_models_dir = common.get_downloaded_models_dir()
        downloaded_models_path = os.path.join(downloaded_models_dir, model_type)
        logger.info(
            f"Searching for model card in downloaded_models_path: {downloaded_models_path}"
        )
        model_card_path = file_utilities.find_file_by_regex(
            downloaded_models_path, r".*modelcard\.json$"
        )
        # get the water index from the model card
        water_classes_indices = get_indices_of_classnames(
            model_card_path, ["water", "whitewater"]
        )
        # Sample class mapping {0:'water',  1:'whitewater', 2:'sand', 3:'rock'}
        class_mapping = get_class_mapping(model_card_path)

        # get the reference shoreline
        reference_shoreline = get_reference_shoreline(
            shoreline, settings["output_epsg"]
        )
        # Add reference shoreline to shoreline_settings
        self.shoreline_settings = self.create_shoreline_settings(
            settings, roi_settings, reference_shoreline
        )
        # Log all items except 'reference shoreline' and handle 'reference shoreline' separately
        logger.info(
            "self.shoreline_settings : "
            + ", ".join(
                f"{key}: {value}"
                for key, value in settings.items()
                if key != "reference_shoreline"
            )
        )
        # Check and log 'reference_shoreline' if it exists
        ref_sl = self.shoreline_settings.get("reference_shoreline", np.array([]))
        if isinstance(ref_sl, np.ndarray):
            logger.info(f"reference_shoreline.shape: {ref_sl.shape}")
        logger.info(
            f"Number of 'reference_shoreline': {len(ref_sl)} for ROI {roi_id}"
        )
        # gets metadata used to extract shorelines
        metadata = get_metadata(self.shoreline_settings["inputs"])
        sitename = self.shoreline_settings["inputs"]["sitename"]
        filepath_data = self.shoreline_settings["inputs"]["filepath"]

        # filter out files that were removed from RGB directory
        try:
            metadata = common.filter_metadata(metadata, sitename, filepath_data)
        except FileNotFoundError as e:
            logger.warning(f"No RGB files existed so no metadata.")
            self.dictionary = {}
            return self
        else:
            # Log portions of the metadata because is massive
            for satname in metadata.keys():
                if not metadata[satname]:
                    logger.warning(f"metadata['{satname}'] is empty")
                else:
                    logger.info(
                        f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('epsg',[]))} of epsg: {np.unique(metadata[satname].get('epsg',[]))}"
                    )
                    logger.info(
                        f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('dates',[]))} of dates Sample first five: {list(islice(metadata[satname].get('dates',[]),5))}"
                    )
                    logger.info(
                        f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('filenames',[]))} of filenames Sample first five: {list(islice(metadata[satname].get('filenames',[]),5))}"
                    )
                    logger.info(
                        f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('im_dimensions',[]))} of im_dimensions: {np.unique(metadata[satname].get('im_dimensions',[]))}"
                    )
                    logger.info(
                        f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('acc_georef',[]))} of acc_georef: {np.unique(metadata[satname].get('acc_georef',[]))}"
                    )
                    logger.info(
                        f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('im_quality',[]))} of im_quality: {np.unique(metadata[satname].get('im_quality',[]))}"
                    )

            extracted_shorelines_dict = extract_shorelines_with_dask(
                session_path,
                metadata,
                self.shoreline_settings,
                class_indices=water_classes_indices,
                class_mapping=class_mapping,
                save_location=new_session_path,
                shoreline_extraction_area=shoreline_extraction_area,
            )
            if extracted_shorelines_dict == {}:
                raise Exception(f"Failed to extract any shorelines.")

            # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
            extracted_shorelines_dict = remove_duplicates(
                extracted_shorelines_dict
            )  # removes duplicates (images taken on the same date by the same satellite
            extracted_shorelines_dict = remove_inaccurate_georef(
                extracted_shorelines_dict, 10
            )  # remove inaccurate georeferencing (set threshold to 10 m)

            # Check and log 'reference shoreline' if it exists
            shorelines_array = extracted_shorelines_dict.get("shorelines", np.array([]))
            if isinstance(shorelines_array, np.ndarray):
                logger.info(f"shorelines.shape: {shorelines_array.shape}")
            logger.info(f"Number of 'shorelines': {len(shorelines_array)}")

            logger.info(
                f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('dates',[]))} of dates: {list(islice(extracted_shorelines_dict.get('dates',[]),3))}"
            )
            logger.info(
                f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('satname',[]))} of satname: {np.unique(extracted_shorelines_dict.get('satname',[]))}"
            )
            logger.info(
                f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('geoaccuracy',[]))} of geoaccuracy: {np.unique(extracted_shorelines_dict.get('geoaccuracy',[]))}"
            )
            logger.info(
                f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('cloud_cover',[]))} of cloud_cover: {np.unique(extracted_shorelines_dict.get('cloud_cover',[]))}"
            )
            logger.info(
                f"extracted_shorelines_dict length {len(extracted_shorelines_dict.get('filename',[]))} of filename[:3]: {list(islice(extracted_shorelines_dict.get('filename',[]),3))}"
            )

            self.dictionary = extracted_shorelines_dict

            if is_list_empty(self.dictionary["shorelines"]):
                logger.warning(f"No extracted shorelines for ROI {roi_id}")
                raise exceptions.No_Extracted_Shoreline(roi_id)

            # # extracted shorelines have map crs so they can be displayed on the map
            # self.gdf = self.create_geodataframe(
            #     self.shoreline_settings["output_epsg"], output_crs="EPSG:4326"
            # )
        return self

    def _validate_input_params(
        self,
        roi_id: str,
        shoreline: gpd.GeoDataFrame,
        roi_settings: dict,
        settings: dict,
    ) -> None:
        """
        Validates that the input parameters for shoreline extraction are not empty and are of the correct type.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): A dictionary of region of interest settings.
        - settings (dict): A dictionary of extraction settings.

        Raises:
        - ValueError: If any of the input parameters are empty or not of the correct type.
        """
        if not isinstance(roi_id, str):
            raise ValueError(f"ROI id must be string. not {type(roi_id)}")

        if not isinstance(shoreline, gpd.GeoDataFrame):
            raise ValueError(
                f"shoreline must be valid geodataframe. not {type(shoreline)}"
            )
        if shoreline.empty:
            raise ValueError("shoreline cannot be empty.")

        if not isinstance(roi_settings, dict):
            raise ValueError(f"roi_settings must be dict. not {type(roi_settings)}")
        if roi_settings == {}:
            raise ValueError("roi_settings cannot be empty.")

        if not isinstance(settings, dict):
            raise ValueError(f"settings must be dict. not {type(settings)}")
        if settings == {}:
            raise ValueError("settings cannot be empty.")

    def extract_shorelines(
            self,
            shoreline_gdf: gpd.geodataframe,
            roi_settings: dict,
            settings: dict,
            output_directory: str = None, 
            shoreline_extraction_area : gpd.geodataframe = None           
        ) -> dict:
        """
        Extracts shorelines for a specified region of interest (ROI).
        Args:
            shoreline_gdf (gpd.geodataframe): GeoDataFrame containing the shoreline data.
            roi_settings (dict): Dictionary containing settings for the ROI. It must have the following keys:
            {
                "dates": ["2018-12-01", "2019-03-01"],
                "sat_list": ["L8", "L9", "S2"],
                "roi_id": "lyw1",
                "polygon": [
                [
                    [-73.94584118213996, 40.57245559853209],
                    [-73.94584118213996, 40.52844804565595],
                    [-73.87282173497694, 40.52844804565595],
                    [-73.87282173497694, 40.57245559853209],
                    [-73.94584118213996, 40.57245559853209]
                ]
                ],
                "landsat_collection": "C02",
                "sitename": "ID_lyw1_datetime01-18-24__12_26_51",
                "filepath": "C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
            },
            settings (dict): Dictionary containing general settings.
            
            session_path (str, optional): Path to the session. Defaults to None.
            class_indices (list, optional): List of class indices. Defaults to None.
            class_mapping (dict, optional): Dictionary mapping class indices to class labels. Defaults to None.
            output_directory (str): The path to the directory where the extracted shorelines will be saved.
                - detection figures will be saved in a subfolder called 'jpg_files' within the output_directory.
                - extract_shoreline reports will be saved within the output_directory.
            shoreline_extraction_area (gpd.geodataframe, optional): A GeoDataFrame containing the area to extract shorelines from. Defaults to None.
        Returns:
            dict: Dictionary containing the extracted shorelines for the specified ROI.
        """
        # project shorelines's crs from map's crs to output crs given in settings
        # create a reference shoreline as a numpy array containing lat, lon, and mean sea level for each point
        reference_shoreline = get_reference_shoreline(
            shoreline_gdf, settings["output_epsg"]
        )
        # Add reference shoreline to shoreline_settings
        self.shoreline_settings = self.create_shoreline_settings(
            settings, roi_settings, reference_shoreline
        )
        # gets metadata used to extract shorelines
        metadata = get_metadata(self.shoreline_settings["inputs"])
        sitename = self.shoreline_settings["inputs"]["sitename"]
        filepath_data = self.shoreline_settings["inputs"]["filepath"]

        # filter out files that were removed from RGB directory
        try:
            metadata = common.filter_metadata(metadata, sitename, filepath_data)
        except FileNotFoundError as e:
            logger.warning(f"No RGB files existed so no metadata.")
            print(
                f"No shorelines were extracted because no RGB files were found at {os.path.join(filepath_data,sitename)}"
            )
            return {}

        for satname in metadata.keys():
            if not metadata[satname]:
                logger.warning(f"metadata['{satname}'] is empty")
            else:
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('epsg',[]))} of epsg: {np.unique(metadata[satname].get('epsg',[]))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('dates',[]))} of dates Sample first five: {list(islice(metadata[satname].get('dates',[]),5))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('filenames',[]))} of filenames Sample first five: {list(islice(metadata[satname].get('filenames',[]),5))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('im_dimensions',[]))} of im_dimensions: {np.unique(metadata[satname].get('im_dimensions',[]))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('acc_georef',[]))} of acc_georef: {np.unique(metadata[satname].get('acc_georef',[]))}"
                )
                logger.info(
                    f"edit_metadata metadata['{satname}'] length {len(metadata[satname].get('im_quality',[]))} of im_quality: {np.unique(metadata[satname].get('im_quality',[]))}"
                )

        # extract shorelines with coastsat's models
        extracted_shorelines = extract_shorelines_coastsat(metadata, self.shoreline_settings,output_directory=output_directory, shoreline_extraction_area=shoreline_extraction_area)
        logger.info(f"extracted_shoreline_dict: {extracted_shorelines}")
        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        extracted_shorelines = remove_duplicates(
            extracted_shorelines
        )  # removes duplicates (images taken on the same date by the same satellite)
        extracted_shorelines = remove_inaccurate_georef(
            extracted_shorelines, 10
        )  # remove inaccurate georeferencing (set threshold to 10 m)
        return extracted_shorelines

    def create_shoreline_settings(
        self,
        settings: dict,
        roi_settings: dict,
        reference_shoreline: dict,
    ) -> None:
        """Create and return a dictionary containing settings for shoreline.
        

        Args:
            settings (dict): settings used to control how shorelines are extracted
            settings = {
                
            "cloud_thresh" (float): percentage of cloud cover allowed
            "cloud_mask_issue" (bool): whether to apply coastsat fix for incorrect cloud masking
            "min_beach_area" (float): minimum area of beach allowed
            "min_length_sl" (int): minimum length (m) of shoreline allowed
            "output_epsg" (int): coordinate reference system of output
            "sand_color" (str): color of sand in RGB image
            "pan_off" (bool): whether to use panchromatic band (always False)
            "max_dist_ref" (int): maximum distance (m) from reference shoreline
            "dist_clouds" (int): distance (m) from clouds to remove
            "percent_no_data" (float): percentage of no data allowed
            "model_session_path" (str): path to model session file
            "apply_cloud_mask" (bool): whether to apply cloud mask
            }
            roi_settings (dict): Dictionary containing settings for the ROI. 
            It must have the following keys:
            {
                "dates": ["2018-12-01", "2019-03-01"],
                "sat_list": ["L8", "L9", "S2"],
                "roi_id": "lyw1",
                "polygon": [
                [
                    [-73.94584118213996, 40.57245559853209],
                    [-73.94584118213996, 40.52844804565595],
                    [-73.87282173497694, 40.52844804565595],
                    [-73.87282173497694, 40.57245559853209],
                    [-73.94584118213996, 40.57245559853209]
                ]
                ],
                "landsat_collection": "C02",
                "sitename": "ID_lyw1_datetime01-18-24__12_26_51",
                "filepath": "C:\\development\\doodleverse\\coastseg\\CoastSeg\\data",
            },
            reference_shoreline (dict): reference shoreline

        Example)
        shoreline_settings =
        {
            "reference_shoreline":reference_shoreline,
            "inputs": roi_settings,
            "adjust_detection": False,
            "check_detection": False,
            ...
            rest of items from settings
        }

        Returns:
            dict: The created shoreline settings.
        """
        SHORELINE_KEYS = [
            "cloud_thresh",
            "cloud_mask_issue",
            "min_beach_area",
            "min_length_sl",
            "output_epsg",
            "sand_color",
            "pan_off",
            "max_dist_ref",
            "dist_clouds",
            "percent_no_data",
            "model_session_path",  # path to model session file
            "apply_cloud_mask",
        ]
        shoreline_settings = {k: v for k, v in settings.items() if k in SHORELINE_KEYS}
        shoreline_settings.update(
            {
                "reference_shoreline": reference_shoreline,
                "adjust_detection": False,  # disable adjusting shorelines manually
                "check_detection": False,  # disable adjusting shorelines manually
                "save_figure": True,  # always save a matplotlib figure of shorelines
                "inputs": roi_settings,  # copy settings for ROI shoreline will be extracted from
            }
        )
        return shoreline_settings

    def create_geodataframe(
        self, input_crs: str, output_crs: str = None, geomtype: str = "lines"
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by input_crs. Converts geodataframe crs
        to output_crs if provided.
        
        Converts the internal dictionary of extracted shorelines to a geodataframe and returns it.
        
        Args:
            input_crs (str ): coordinate reference system string. Format 'EPSG:4326'.
            output_crs (str, optional): coordinate reference system string. Defaults to None.
        Returns:
            gpd.GeoDataFrame: geodataframe with columns = ['geometery','date','satname','geoaccuracy','cloud_cover']
            converted to output_crs if provided otherwise geodataframe's crs will be
            input_crs
        """
        extract_shoreline_gdf = output_to_gdf(self.dictionary, geomtype)
        if not extract_shoreline_gdf.crs:
            extract_shoreline_gdf.set_crs(input_crs, inplace=True)
        if output_crs is not None:
            extract_shoreline_gdf = extract_shoreline_gdf.to_crs(output_crs)
        return extract_shoreline_gdf

    def to_file(
        self, filepath: str, filename: str, data: Union[gpd.GeoDataFrame, dict]
    ):
        """Save geopandas dataframe to file, or save data to file with to_file().

        Args:
            filepath (str): The directory where the file should be saved.
            filename (str): The name of the file to be saved.
            data (Any): The data to be saved to file.

        Raises:
            ValueError: Raised when data is not a geopandas dataframe and cannot be saved with tofile()
        """
        file_location = os.path.abspath(os.path.join(filepath, filename))

        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(
                file_location,
                driver="GeoJSON",
                encoding="utf-8",
            )
        elif isinstance(data, dict):
            if data != {}:
                file_utilities.to_file(data, file_location)

    def get_layer_name(self) -> list:
        """returns name of extracted shoreline layer"""
        layer_name = "extracted_shoreline"
        return layer_name

    def get_styled_layer(
        self, gdf, row_number: int = 0, map_crs: int = 4326, style: dict = {}
    ) -> dict:
        """
        Returns a single shoreline feature as a GeoJSON object with a specified style.

        Args:
        - gdf: The input GeoDataFrame.
        - row_number (int): The index of the shoreline feature to select from the GeoDataFrame.
        - map_crs (int): The desired coordinate reference system.
        - style (dict) default {} :
            Additional style attributes to be merged with the default style.

        Returns:
        - dict: A styled GeoJSON feature.
        """
        if gdf.empty:
            return []

        projected_gdf = transform_gdf_to_crs(gdf, map_crs)
        single_shoreline = select_and_stringify(projected_gdf, row_number)
        features_json = convert_gdf_to_json(single_shoreline)
        layer_name = self.get_layer_name()

        # Ensure there are features to process.
        if not features_json.get("features"):
            return []

        styled_feature = style_layer(
            features_json["features"][0], layer_name, "red", style
        )
        return styled_feature


def get_reference_shoreline(
    shoreline_gdf: gpd.geodataframe, output_crs: str
) -> np.ndarray:
    """
    Converts a GeoDataFrame of shoreline features into a numpy array of latitudes, longitudes, and zeroes representing the mean sea level.

    Args:
    - shoreline_gdf (GeoDataFrame): A GeoDataFrame of shoreline features.
    - output_crs (str): The output CRS to which the shoreline features need to be projected.

    Returns:
    - np.ndarray: A numpy array of latitudes, longitudes, and zeroes representing the mean sea level.
    """
    # project shorelines's espg from map's espg to output espg given in settings
    reprojected_shorlines = shoreline_gdf.to_crs(output_crs)
    # convert shoreline_in_roi gdf to coastsat compatible format np.array([[lat,lon,0],[lat,lon,0]...])
    shorelines = make_coastsat_compatible(reprojected_shorlines)
    # shorelines = [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...]
    # Stack all the tuples into a single list of n rows X 2 columns
    shorelines = np.vstack(shorelines)
    # Add third column of 0s to represent mean sea level
    shorelines = np.insert(shorelines, 2, np.zeros(len(shorelines)), axis=1)
    
    return shorelines


def get_colors(length: int) -> list:
    # returns a list of color hex codes as long as length
    cmap = get_cmap("plasma", length)
    cmap_list = [rgb2hex(i) for i in cmap.colors]
    return cmap_list


def make_coastsat_compatible(feature: gpd.geodataframe) -> list:
    """Return the feature as an np.array in the form:
        [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    Args:
        feature (gpd.geodataframe): clipped portion of shoreline within a roi
    Returns:
        list: shorelines in form:
            [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    """
    features = []
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode(index_parts=True)
    # For each linestring portion of feature convert to lat,lon tuples
    lat_lng = feature_exploded.apply(
        lambda row: tuple(np.array(row.geometry.coords).tolist()), axis=1
    )
    features = list(lat_lng)
    return features


def is_list_empty(main_list: list) -> bool:
    all_empty = True
    for np_array in main_list:
        if len(np_array) != 0:
            all_empty = False
    return all_empty
