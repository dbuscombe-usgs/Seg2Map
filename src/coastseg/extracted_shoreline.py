# Standard library imports
import colorsys
import copy
import fnmatch
import json
import json
import logging
import os
from glob import glob
from time import perf_counter
from typing import Optional, Union, List, Dict
from time import perf_counter
from typing import Dict, List, Optional, Union

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
from coastsat import SDS_preprocess, SDS_shoreline, SDS_tools
from coastsat.SDS_download import get_metadata
from coastsat.SDS_shoreline import extract_shorelines
from coastsat.SDS_tools import (
    get_filenames,
    get_filepath,
    output_to_gdf,
    remove_duplicates,
    remove_inaccurate_georef,
)
from coastsat.SDS_transects import compute_intersection_QC
from ipyleaflet import GeoJSON
from matplotlib import gridspec
from matplotlib.colors import rgb2hex
from matplotlib.pyplot import get_cmap
from skimage import measure, morphology
from tqdm.auto import tqdm

# Internal dependencies imports
from coastseg import common, exceptions
from coastseg.validation import get_satellites_in_directory
from coastseg.filters import filter_model_outputs
from coastseg.common import get_filtered_files_dict, edit_metadata


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
        logger.debug(f"{func.__name__} took {end - start:.6f} seconds to run.")
        return result

    return wrapper


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
    logger.info(f"transects.crs: {transects_gdf.crs} transects: {transects_gdf}")
    transects = common.get_transect_points_dict(transects_gdf)
    logger.info(f"transects as dictionary for coastsat: {transects}")
    # print(f'settings to extract transects: {settings}')
    # cross_distance: along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
    cross_distance = compute_intersection_QC(extracted_shorelines, transects, settings)
    return cross_distance


def combine_satellite_data(satellite_data: dict):
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
    merged_satellite_data = {}
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
        # For each key in the nested dictionary
        for key, value in sat_data.items():
            # Wrap non-list values in a list and concatenate to merged_satellite_data
            if not isinstance(value, list):
                merged_satellite_data[key] += [value]
            else:
                merged_satellite_data[key] += value
        # Add satellite name entries for each date
        if "dates" in sat_data.keys():
            merged_satellite_data["satname"] += [satname] * len(sat_data["dates"])
    # Sort chronologically
    idx_sorted = sorted(
        range(len(merged_satellite_data["dates"])),
        key=lambda i: merged_satellite_data["dates"][i],
    )

    for key in merged_satellite_data.keys():
        merged_satellite_data[key] = [merged_satellite_data[key][i] for i in idx_sorted]

    return merged_satellite_data


def process_satellite(
    satname: str,
    settings: dict,
    metadata: dict,
    session_path: str,
    class_indices: list,
    class_mapping: dict,
    save_location: str,
):
    # set batch size
    batch_size = 10
    logger.info(f"metadata: {metadata}")
    # filenames of tifs (ms) for this satellite
    filenames = metadata[satname]["filenames"]
    output = {}
    if len(filenames) == 0:
        logger.warning(f"Satellite {satname} had no imagery")
        return output

    collection = settings["inputs"]["landsat_collection"]
    default_min_length_sl = settings["min_length_sl"]
    # deep copy settings
    settings = copy.deepcopy(settings)
    filepath = get_filepath(settings["inputs"], satname)
    pixel_size = get_pixel_size_for_satellite(satname)

    # get the minimum beach area in number of pixels depending on the satellite
    settings["min_length_sl"] = get_min_shoreline_length(satname, default_min_length_sl)

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
            espg_list.append(metadata[satname]["epsg"][index])
            geoaccuracy_list.append(metadata[satname]["acc_georef"][index])
            timestamps.append(metadata[satname]["dates"][index])

            tasks.append(
                dask.delayed(process_satellite_image)(
                    filenames[index],
                    filepath,
                    settings,
                    satname,
                    collection,
                    image_epsg,
                    pixel_size,
                    session_path,
                    class_indices,
                    class_mapping,
                    save_location,
                )
            )

        # compute tasks in batches
        results = dask.compute(*tasks)
        # update progress bar
        num_tasks_computed = len(tasks)
        pbar.update(num_tasks_computed)

        for index, result in enumerate(results):
            if result is None:
                continue
            output.setdefault(satname, {})
            output[satname].setdefault("dates", []).append(timestamps[index])
            output[satname].setdefault("geoaccuracy", []).append(
                geoaccuracy_list[index]
            )
            output[satname].setdefault("shorelines", []).append(result["shorelines"])
            output[satname].setdefault("cloud_cover", []).append(result["cloud_cover"])
            output[satname].setdefault("filename", []).append(filenames[index])
            output[satname].setdefault("idx", []).append(index)

    pbar.close()
    return output


def get_cloud_cover_combined(cloud_mask: np.ndarray):
    """
    Calculate the cloud cover percentage (ignoring no-data pixels) of a cloud_mask.

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


def process_satellite_image(
    filename: str,
    filepath,
    settings,
    satname,
    collection,
    image_epsg,
    pixel_size,
    session_path,
    class_indices,
    class_mapping,
    save_location: str,
):
    # get image date
    date = filename[:19]
    # get image filename
    fn = get_filenames(filename, filepath, satname)
    # preprocess image (cloud mask + pansharpening/downsampling)
    # could apply dask delayed here
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
        do_cloud_mask=settings.get("apply_cloud_mask", True),
    )

    logger.info(f"process_satellite_image_settings: {settings}")
    # if percentage of no data pixels are greater than allowed, skip
    percent_no_data_allowed = settings.get("percent_no_data", None)
    logger.info(f"percent_no_data_allowed: {percent_no_data_allowed}")
    if percent_no_data_allowed is not None:
        percent_no_data_allowed = percent_no_data_allowed / 100
        num_total_pixels = cloud_mask.shape[0] * cloud_mask.shape[1]
        percentage_no_data = np.sum(im_nodata) / num_total_pixels
        logger.info(f"percentage_no_data: {percentage_no_data}")
        logger.info(f"percent_no_data_allowed: {percent_no_data_allowed}")
        if percentage_no_data > percent_no_data_allowed:
            logger.info(
                f"percent_no_data_allowed exceeded {percentage_no_data} > {percent_no_data_allowed}"
            )
            return None

    # compute cloud_cover percentage (with no data pixels)
    cloud_cover_combined = get_cloud_cover_combined(cloud_mask)
    if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
        logger.info("cloud_cover_combined > 0.99")
        return None
    # Remove no data pixels from the cloud mask
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
    # compute cloud cover percentage (without no data pixels)
    cloud_cover = get_cloud_cover(cloud_mask, im_nodata)
    # skip image if cloud cover is above user-defined threshold
    if cloud_cover > settings["cloud_thresh"]:
        logger.info("Cloud thresh exceeded")
        return None
    # calculate a buffer around the reference shoreline (if any has been digitised)
    ref_shoreline_buffer = SDS_shoreline.create_shoreline_buffer(
        cloud_mask.shape, georef, image_epsg, pixel_size, settings
    )
    # read the model outputs from the npz file for this image
    npz_file = find_matching_npz(filename, os.path.join(session_path, "good"))
    if npz_file is None:
        npz_file = find_matching_npz(filename, session_path)
    logger.info(f"npz_file: {npz_file}")
    if npz_file is None:
        logger.warning(f"npz file not found for {filename}")
        return None

    # get the labels for water and land
    merged_labels = load_merged_image_labels(npz_file, class_indices=class_indices)
    all_labels = load_image_labels(npz_file)

    min_beach_area = settings["min_beach_area"]
    # bad idea to use remove_small_objects_and_binarize on all_labels, safe to use on merged_labels (water/land boundary)
    # all_labels = morphology.remove_small_objects(all_labels, min_size=min_beach_area, connectivity=2)
    merged_labels = remove_small_objects_and_binarize(merged_labels, min_beach_area)

    logger.info(f"merged_labels: {merged_labels}\n")
    if sum(merged_labels[ref_shoreline_buffer]) < 50:
        logger.warning(
            f"{fn} Not enough sand pixels within the beach buffer to detect shoreline"
        )
        return None
    # get the shoreline from the image
    shoreline = find_shoreline(
        fn,
        image_epsg,
        settings,
        cloud_mask_adv,
        cloud_mask,
        im_nodata,
        georef,
        merged_labels,
        ref_shoreline_buffer,
    )
    if shoreline is None:
        logger.warning(f"\nShoreline not found for {fn}")
        return None
    # plot the results
    if not settings["check_detection"]:
        plt.ioff()
    shoreline_detection_figures(
        im_ms,
        cloud_mask,
        merged_labels,
        all_labels,
        shoreline,
        image_epsg,
        georef,
        settings,
        date,
        satname,
        class_mapping,
        save_location,
    )
    # create dictionnary of output
    output = {
        "shorelines": shoreline,
        "cloud_cover": cloud_cover,
    }
    logger.info(f"output shorelines and cloud cover: {output}")
    return output


def get_model_card_classes(model_card_path: str) -> dict:
    """return the classes dictionary from the model card
        example classes dictionary {0: 'sand', 1: 'sediment', 2: 'whitewater', 3: 'water'}
    Args:
        model_card_path (str): path to model card

    Returns:
        dict: dictionary of classes in model card and their corresponding index
    """
    model_card_data = file_utilities.read_json_file(model_card_path, raise_error=True)
    logger.info(
        f"model_card_path: {model_card_path} \nmodel_card_data: {model_card_data}"
    )
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
    merged_legend,
    all_legend,
    titles: list[str] = [],
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
    ax1.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=3)
    ax1.set_title(titles[0])
    ax1.axis("off")

    # Plot first combined image with overlay and legend
    ax2.imshow(merged_overlay)
    ax2.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=3)
    ax2.set_title(titles[1])
    ax2.axis("off")
    if merged_legend:  # Check if the list is not empty
        ax2.legend(
            handles=merged_legend,
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            borderaxespad=0.0,
        )

    # Plot second combined image with overlay and legend
    ax3.imshow(all_overlay)
    ax3.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=3)
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
    fig.savefig(os.path.join(filepath, date + "_" + satname + ".jpg"), dpi=150)
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
    """
    sitename = settings["inputs"]["sitename"]
    if save_location:
        filepath = os.path.join(save_location, "jpg_files", "detection")
    else:
        filepath_data = settings["inputs"]["filepath"]
        filepath = os.path.join(filepath_data, sitename, "jpg_files", "detection")
    os.makedirs(filepath, exist_ok=True)
    logger.info(f"shoreline_detection_figures filepath: {filepath}")

    # increase the intensity of the image for visualization
    im_RGB = increase_image_intensity(im_ms, cloud_mask, prob_high=99.9)
    logger.info(
        f"im_RGB.shape: {im_RGB.shape}\n im_RGB.dtype: {im_RGB.dtype}\n im_RGB: {np.unique(im_RGB)[:5]}\n"
    )

    im_merged = create_overlay(im_RGB, merged_labels, overlay_opacity=0.35)
    im_all = create_overlay(im_RGB, all_labels, overlay_opacity=0.35)

    logger.info(
        f"im_merged.shape: {im_merged.shape}\n im_merged.dtype: {im_merged.dtype}\n im_merged.max: {im_merged.max()}\n im_merged.min: {im_merged.min()}\n"
    )
    logger.info(
        f"im_all.shape: {im_all.shape}\n im_all.dtype: {im_all.dtype}\n im_all: {np.unique(im_all)[:5]}\n"
    )

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

    # Create legend for the shorelines
    black_line = mlines.Line2D([], [], color="k", linestyle="-", label="shoreline")

    # create a legend for the class colors and the shoreline
    all_classes_legend = create_legend(class_mapping, additional_patches=[black_line])
    merged_classes_legend = create_legend(
        class_mapping={0: "other", 1: "water"}, additional_patches=[black_line]
    )

    # Plot images
    fig = plot_image_with_legend(
        im_RGB,
        im_merged,
        im_all,
        pixelated_shoreline,
        merged_classes_legend,
        all_classes_legend,
        titles=[sitename, date, satname],
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
    # Apply the cloud mask by setting masked pixels to a special value (e.g., -1)
    im_labels_masked = im_labels.copy()
    # im_labels_masked[cloud_mask] = -1
    # # Apply the reference shoreline buffer mask by setting masked pixels to a special value (e.g., -1)
    # im_labels_masked[~reference_shoreline_buffer] = -1
    im_labels_masked[cloud_mask] = np.NaN
    # Apply the reference shoreline buffer mask by setting masked pixels to a special value (e.g., -1)
    im_labels_masked[~reference_shoreline_buffer] = np.NaN

    # 0 or 1 labels means 0.5 is the threshold
    contours = measure.find_contours(im_labels_masked, 0.5)

    # remove contour points that are NaNs (around clouds)
    processed_contours = SDS_shoreline.process_contours(contours)

    return processed_contours


def find_shoreline(
    fn,
    image_epsg,
    settings,
    cloud_mask_adv,
    cloud_mask,
    im_nodata,
    georef,
    im_labels,
    reference_shoreline_buffer,
) -> np.array:
    try:
        contours = simplified_find_contours(
            im_labels, cloud_mask, reference_shoreline_buffer
        )
    except Exception as e:
        logger.error(f"{e}\nCould not map shoreline for this image: {fn}")
        return None
    # print(f"Settings used by process_shoreline: {settings}")
    # process the water contours into a shoreline
    shoreline = SDS_shoreline.process_shoreline(
        contours, cloud_mask_adv, im_nodata, georef, image_epsg, settings
    )
    return shoreline


@time_func
def extract_shorelines_with_dask(
    session_path: str,
    metadata: dict,
    settings: dict,
    class_indices: list = None,
    class_mapping: dict = None,
    save_location: str = "",
) -> dict:
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    # initialise output structure
    extracted_shorelines_data = {}
    if not save_location:
        # create a subfolder to store the .jpg images showing the detection
        filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
        os.makedirs(filepath_jpg, exist_ok=True)

    # loop through satellite list
    # filenames = metadata[satname]["filenames"]
    # output = {}
    # if len(filenames) == 0:
    #     logger.warning(f"Satellite {satname} had no imagery")
    #     return output
    good_folder = os.path.join(session_path, "good")
    bad_folder = os.path.join(session_path, "bad")
    satellites = get_satellites_in_directory(session_path)
    # for each satellite sort the model outputs into good & bad
    for satname in satellites:
        # get all the model_outputs that have the satellite in the filename
        files = glob(f"{session_path}{os.sep}*{satname}*.npz")
        if len(files) != 0:
            filter_model_outputs(satname, files, good_folder, bad_folder)

    # for each satellite get the list of files that were sorted as 'good'
    filtered_files = get_filtered_files_dict(good_folder, "npz", sitename)
    # keep only the metadata for the files that were sorted as 'good'
    metadata = edit_metadata(metadata, filtered_files)

    result_dict = {}
    for satname in metadata:
        satellite_dict = process_satellite(
            satname,
            settings,
            metadata,
            session_path,
            class_indices,
            class_mapping,
            save_location,
        )
        result_dict.update(satellite_dict)
    extracted_shorelines_data = combine_satellite_data(result_dict)
    logger.info(f"extracted_shorelines_data: {extracted_shorelines_data}")
    return extracted_shorelines_data


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
    extracted_shorelines = extracted_shorelines.load_extracted_shorelines(
        extracted_files["dict"], extracted_files["settings"], extracted_files["geojson"]
    )

    logger.info(f"Loaded extracted shorelines from: {dir_path}")
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
        return f"Extracted Shoreline: ROI ID: {self.roi_id}\n geodataframe {self.gdf.head(5)}\nshoreline_settings{self.shoreline_settings}"

    def __repr__(self):
        return f"Extracted Shoreline:  ROI ID: {self.roi_id}\n geodataframe {self.gdf.head(5)}\nshoreline_settings{self.shoreline_settings}\ndictionary{self.dictionary}"

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
    ) -> "Extracted_Shoreline":
        """
        Extracts shorelines for a specified region of interest (ROI) and returns an Extracted_Shoreline class instance.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): A dictionary of region of interest settings.
        - settings (dict): A dictionary of extraction settings.

        Returns:
        - object: The Extracted_Shoreline class instance.
        """
        # validate input parameters are not empty and are of the correct type
        self._validate_input_params(roi_id, shoreline, roi_settings, settings)

        logger.info(f"Extracting shorelines for ROI id{roi_id}")
        self.dictionary = self.extract_shorelines(
            shoreline,
            roi_settings,
            settings,
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
    ) -> "Extracted_Shoreline":
        """
        Extracts shorelines for a specified region of interest (ROI) from a saved session and returns an Extracted_Shoreline class instance.

        Args:
        - self: The object instance.
        - roi_id (str): The ID of the region of interest for which shorelines need to be extracted.
        - shoreline (GeoDataFrame): A GeoDataFrame of shoreline features.
        - roi_settings (dict): A dictionary of region of interest settings.
        - settings (dict): A dictionary of extraction settings.
        - session_path (str): The path of the saved session from which the shoreline extraction needs to be resumed.
        - new_session_path (str) :The path of the new session where the extreacted shorelines extraction will be saved
        Returns:
        - object: The Extracted_Shoreline class instance.
        """
        # validate input parameters are not empty and are of the correct type
        self._validate_input_params(roi_id, shoreline, roi_settings, settings)

        logger.info(f"Extracting shorelines for ROI id{roi_id}")

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
        # Sample class mapping {0:'water',   1:'whitewater', 2:'sand', 3:'rock'}
        class_mapping = get_class_mapping(model_card_path)

        # get the reference shoreline
        reference_shoreline = get_reference_shoreline(
            shoreline, settings["output_epsg"]
        )
        # Add reference shoreline to shoreline_settings
        self.shoreline_settings = self.create_shoreline_settings(
            settings, roi_settings, reference_shoreline
        )
        logger.info(f"self.shoreline_settings: {self.shoreline_settings}")
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
        else:
            logger.info(f"metadata: {metadata}")

            # self.dictionary = self.extract_shorelines(
            #     shoreline,
            #     roi_settings,
            #     settings,
            #     session_path=session_path,
            #     class_indices=water_classes_indices,
            #     class_mapping=class_mapping,
            # )

            extracted_shorelines_dict = extract_shorelines_with_dask(
                session_path,
                metadata,
                self.shoreline_settings,
                class_indices=water_classes_indices,
                class_mapping=class_mapping,
                save_location=new_session_path,
            )
            if extracted_shorelines_dict == {}:
                raise Exception(f"Failed to extract any shorelines.")

            logger.info(f"extracted_shoreline_dict: {extracted_shorelines_dict}")
            # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
            extracted_shorelines_dict = remove_duplicates(
                extracted_shorelines_dict
            )  # removes duplicates (images taken on the same date by the same satellite)
            extracted_shorelines_dict = remove_inaccurate_georef(
                extracted_shorelines_dict, 10
            )  # remove inaccurate georeferencing (set threshold to 10 m)
            logger.info(
                f"after remove_inaccurate_georef : extracted_shoreline_dict: {extracted_shorelines_dict}"
            )
            self.dictionary = extracted_shorelines_dict

            if is_list_empty(self.dictionary["shorelines"]):
                logger.warning(f"No extracted shorelines for ROI {roi_id}")
                raise exceptions.No_Extracted_Shoreline(roi_id)

            map_crs = "EPSG:4326"
            # extracted shorelines have map crs so they can be displayed on the map
            self.gdf = self.create_geodataframe(
                self.shoreline_settings["output_epsg"], output_crs=map_crs
            )
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
        shoreline_gdf: gpd.GeoDataFrame,
        roi_settings: dict,
        settings: dict,
        session_path: str = None,
        class_indices: list = None,
        class_mapping: dict = None,
    ) -> dict:
        """Returns a dictionary containing the extracted shorelines for roi specified by rois_gdf"""
        # project shorelines's crs from map's crs to output crs given in settings
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
            logger.info(f"metadata before filter : {metadata}")
            metadata = common.filter_metadata(metadata, sitename, filepath_data)
            logger.info(f"metadata after filter : {metadata}")
        except FileNotFoundError as e:
            logger.warning(f"No RGB files existed so no metadata.")
            return {}

        logger.info(f"new metadata: {metadata}")
        logger.info(f"self.shoreline_settings: {self.shoreline_settings}")
        # extract shorelines from ROI
        if session_path is None:
            # extract shorelines with coastsat's models
            extracted_shorelines = extract_shorelines(metadata, self.shoreline_settings)
        elif session_path is not None:
            # extract shorelines with our models
            extracted_shorelines = extract_shorelines_with_dask(
                session_path,
                metadata,
                self.shoreline_settings,
                class_indices=class_indices,
                class_mapping=class_mapping,
            )
        logger.info(f"extracted_shoreline_dict: {extracted_shorelines}")
        # postprocessing by removing duplicates and removing in inaccurate georeferencing (set threshold to 10 m)
        extracted_shorelines = remove_duplicates(
            extracted_shorelines
        )  # removes duplicates (images taken on the same date by the same satellite)
        extracted_shorelines = remove_inaccurate_georef(
            extracted_shorelines, 10
        )  # remove inaccurate georeferencing (set threshold to 10 m)
        logger.info(
            f"after remove_inaccurate_georef : extracted_shoreline_dict: {extracted_shorelines}"
        )
        return extracted_shorelines

    def create_shoreline_settings(
        self,
        settings: dict,
        roi_settings: dict,
        reference_shoreline: dict,
    ) -> None:
        """Create and return a dictionary containing settings for shoreline.

        Args:
            settings (dict): map settings
            roi_settings (dict): settings of the roi. Must include 'dates'
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
        logger.info(f"settings used to create shoreline settings: {settings}")
        shoreline_settings = {k: v for k, v in settings.items() if k in SHORELINE_KEYS}
        logger.info(f"Loading shoreline_settings: {shoreline_settings}")

        shoreline_settings.update(
            {
                "reference_shoreline": reference_shoreline,
                "adjust_detection": False,  # disable adjusting shorelines manually
                "check_detection": False,  # disable adjusting shorelines manually
                "save_figure": True,  # always save a matplotlib figure of shorelines
                "inputs": roi_settings,  # copy settings for ROI shoreline will be extracted from
            }
        )

        logger.info(f"shoreline_settings: {shoreline_settings}")
        return shoreline_settings

    def create_geodataframe(
        self, input_crs: str, output_crs: str = None, geomtype: str = "lines"
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by input_crs. Converts geodataframe crs
        to output_crs if provided.
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
        logger.info(
            f"extract_shoreline_gdf inital crs {extract_shoreline_gdf.crs} extract_shoreline_gdf {extract_shoreline_gdf}"
        )
        if output_crs is not None:
            extract_shoreline_gdf = extract_shoreline_gdf.to_crs(output_crs)
        logger.info(
            f"extract_shoreline_gdf final crs {extract_shoreline_gdf.crs} extract_shoreline_gdf {extract_shoreline_gdf}"
        )
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
    shoreline_gdf: gpd.GeoDataFrame, output_crs: str
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
    logger.info(f"reprojected_shorlines.crs: {reprojected_shorlines.crs}")
    logger.info(f"reprojected_shorlines: {reprojected_shorlines}")
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


def make_coastsat_compatible(feature: gpd.GeoDataFrame) -> list:
    """Return the feature as an np.array in the form:
        [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    Args:
        feature (gpd.GeoDataFrame): clipped portion of shoreline within a roi
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
