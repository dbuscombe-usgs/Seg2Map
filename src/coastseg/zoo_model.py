import os

from pathlib import Path
import re
import glob
import asyncio
import platform
import json
import logging
from itertools import islice
from typing import List, Set, Tuple

from coastseg import common
from coastseg import downloads
from coastseg import sessions
from coastseg import extracted_shoreline
from coastseg import geodata_processing
from coastseg import file_utilities
from coastseg import core_utilities

import geopandas as gpd
from osgeo import gdal
import skimage
import aiohttp
import tqdm
from PIL import Image
import numpy as np
from glob import glob
import tqdm.asyncio
import nest_asyncio

from skimage.io import imread
from tensorflow.keras import mixed_precision
from doodleverse_utils.prediction_imports import do_seg
from doodleverse_utils.model_imports import (
    simple_resunet,
    custom_resunet,
    custom_unet,
    simple_unet,
    simple_resunet,
    simple_satunet,
    segformer,
)
from doodleverse_utils.model_imports import dice_coef_loss, iou_multi, dice_multi
# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import tensorflow as tf

logger = logging.getLogger(__name__)


def filter_no_data_pixels(files: list[str], percent_no_data: float = 0.50) -> list[str]:
    def percentage_of_black_pixels(img: "PIL.Image") -> float:
        # Calculate the total number of pixels in the image
        num_total_pixels = img.size[0] * img.size[1]
        img_array = np.array(img)
        # Count the number of black pixels in the image
        black_pixels = np.count_nonzero(np.all(img_array == 0, axis=-1))
        # Calculate the percentage of black pixels
        percentage = (black_pixels / num_total_pixels) 
        return percentage

    valid_images = []

    def has_image_files(file_list, extensions):
        return any(file.lower().endswith(extensions) for file in file_list)

    extensions = (".jpg", ".png", ".jpeg")
    contains_images = has_image_files(files, extensions)
    if not contains_images:
        logger.warning(
            f"Cannot filter no data pixels no images found with {extensions} in {files}."
        )
        return files

    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            img = Image.open(file)
            percentage = percentage_of_black_pixels(img)
            if percentage <= percent_no_data:
                valid_images.append(file)

    return valid_images


def get_files_to_download(
    available_files: List[dict], filenames: List[str], model_id: str, model_path: str
) -> dict:
    """Constructs a dictionary of file paths and their corresponding download links, based on the available files and a list of desired filenames.

    Args:
    - available_files: A list of dictionaries representing the metadata of available files, including the file key and download links.
    - filenames: A list of strings representing the desired filenames.
    - model_id: A string representing the ID of the model being downloaded.
    - model_path: A string representing the path to the directory where the files will be downloaded.

    Returns:
    A dictionary with file paths as keys and their corresponding download links as values.
    Raises a ValueError if any of the desired filenames are not available in the available_files list.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    url_dict = {}
    for filename in filenames:
        response = next((f for f in available_files if f["key"] == filename), None)
        if response is None:
            raise ValueError(f"Cannot find {filename} at {model_id}")
        link = response["links"]["self"]
        file_path = os.path.join(model_path, filename)
        url_dict[file_path] = link
    return url_dict


def check_if_files_exist(files_dict: dict) -> dict:
    """Checks if each file in a given dictionary of file paths and download links already exists in the local filesystem.

    Args:
    - files_dict: A dictionary with file paths as keys and their corresponding download links as values.

    Returns:
    A dictionary with file paths as keys and their corresponding download links as values, for any files that do not exist in the local filesystem.
    """
    url_dict = {}
    for save_path, link in files_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path] = link
    return url_dict


def get_zenodo_release(zenodo_id: str) -> dict:
    """
    Retrieves the JSON data for the Zenodo release with the given ID.
    """
    root_url = f"https://zenodo.org/api/records/{zenodo_id}"
    # get a response from the url
    response = common.get_response(root_url, stream=False)
    response.raise_for_status()
    return response.json()


def get_imagery_directory(img_type: str, RGB_path: str) -> str:
    """
    Returns directory of the newly created imagery. Available imagery conversions:

    1. 'NDWI' for 'NIR'
    2. 'MNDWI' for 'SWIR'
    3. 'RGB+MNDWI+NDWI' for 'RGB','NIR','SWIR'
    4. 'RGB' for 'RGB'

    Note:
        Directories containing 'NIR','NIR' and 'RGB' imagery must be at the same level as the 'RGB' imagery.
        ex.
        home/
            RGB
            NIR
            SWIR

    Args:
        img_type (str): The type of imagery to generate. Available options: 'RGB', 'NDWI', 'MNDWI',or 'RGB+MNDWI+NDWI'
        RGB_path (str): The path to the RGB imagery directory.

    Returns:
        str: The path to the output directory for the specified imagery type.
    """
    img_type = img_type.upper()
    output_path = os.path.dirname(RGB_path)
    if img_type == "RGB":
        output_path = RGB_path
    elif img_type == "RGB+MNDWI+NDWI":
        NIR_path = os.path.join(output_path, "NIR")
        NDWI_path = RGB_to_infrared(RGB_path, NIR_path, output_path, "NDWI")
        SWIR_path = os.path.join(output_path, "SWIR")
        MNDWI_path = RGB_to_infrared(RGB_path, SWIR_path, output_path, "MNDWI")
        five_band_path = file_utilities.create_directory(output_path, "five_band")
        output_path = get_five_band_imagery(
            RGB_path, MNDWI_path, NDWI_path, five_band_path
        )
    # default filetype is NIR and if NDWI is selected else filetype to SWIR
    elif img_type == "NDWI":
        NIR_path = os.path.join(output_path, "NIR")
        output_path = RGB_to_infrared(RGB_path, NIR_path, output_path, "NDWI")
    elif img_type == "MNDWI":
        SWIR_path = os.path.join(output_path, "SWIR")
        output_path = RGB_to_infrared(RGB_path, SWIR_path, output_path, "MNDWI")
    else:
        raise ValueError(
            f"{img_type} not reconigzed as one of the valid types 'RGB', 'NDWI', 'MNDWI',or 'RGB+MNDWI+NDWI'"
        )
    return output_path


def get_five_band_imagery(
    RGB_path: str, MNDWI_path: str, NDWI_path: str, output_path: str
):
    paths = [RGB_path, MNDWI_path, NDWI_path]
    files = []
    for data_path in paths:
        f = sorted(glob(data_path + os.sep + "*.jpg"))
        if len(f) < 1:
            f = sorted(glob(data_path + os.sep + "images" + os.sep + "*.jpg"))
        files.append(f)

    # number of bands x number of samples
    files = np.vstack(files).T
    # returns path to five band imagery
    for counter, file in enumerate(files):
        im = []  # read all images into a list
        for k in file:
            im.append(imread(k))
        datadict = {}
        # create stack which takes care of different sized inputs
        im = np.dstack(im)
        datadict["arr_0"] = im.astype(np.uint8)
        datadict["num_bands"] = im.shape[-1]
        datadict["files"] = [file_name.split(os.sep)[-1] for file_name in file]
        ROOT_STRING = file[0].split(os.sep)[-1].split(".")[0]
        segfile = (
            output_path
            + os.sep
            + ROOT_STRING
            + "_noaug_nd_data_000000"
            + str(counter)
            + ".npz"
        )
        np.savez_compressed(segfile, **datadict)
        del datadict, im
    return output_path


def matching_datetimes_files(dir1: str, dir2: str) -> Set[str]:
    """
    Get the matching datetimes from the filenames in two directories.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.

    Returns:
        Set[str]: A set of strings representing the common datetimes.
    """
    # Get the filenames in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Define a pattern to match the date-time part of the filenames
    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    )  # Matches YYYY-MM-DD-HH-MM-SS

    # Create sets of the date-time parts of the filenames in each directory
    files1_dates = {
        re.search(pattern, filename).group(0)
        for filename in files1
        if re.search(pattern, filename)
    }
    files2_dates = {
        re.search(pattern, filename).group(0)
        for filename in files2
        if re.search(pattern, filename)
    }

    # Find the intersection of the two sets
    matching_files = files1_dates & files2_dates

    return matching_files


def get_full_paths(
    dir1: str, dir2: str, common_dates: Set[str]
) -> Tuple[List[str], List[str]]:
    """
    Get the full paths of the files with matching datetimes.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
        common_dates (Set[str]): A set of strings representing the common datetimes.

    Returns:
        Tuple[List[str], List[str]]: Two lists of strings representing the full paths of the matching files in dir1 and dir2.
    """
    # Get the filenames in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Define a pattern to match the date-time part of the filenames
    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    )  # Matches YYYY-MM-DD-HH-MM-SS

    # Find the full paths of the files with the matching date-times
    matching_files_dir1 = [
        os.path.join(dir1, filename)
        for filename in files1
        if re.search(pattern, filename)
        and re.search(pattern, filename).group(0) in common_dates
    ]
    matching_files_dir2 = [
        os.path.join(dir2, filename)
        for filename in files2
        if re.search(pattern, filename)
        and re.search(pattern, filename).group(0) in common_dates
    ]

    return matching_files_dir1, matching_files_dir2


def get_files(RGB_dir_path: str, img_dir_path: str) -> np.ndarray:
    """returns matrix of files in RGB_dir_path and img_dir_path
    creates matrix: RGB x number of samples in img_dir_path
    Example:
    [['full_RGB_path.jpg','full_NIR_path.jpg'],
    ['full_jpg_path.jpg','full_NIR_path.jpg']....]
    Args:
        RGB_dir_path (str): full path to directory of RGB images
        img_dir_path (str): full path to directory of non-RGB images
        usually NIR and SWIR

    Raises:
        FileNotFoundError: raised if directory is not found
    Returns:
        np.ndarray: A matrix of matching files, shape (bands, number of samples).
    """
    if not os.path.exists(RGB_dir_path):
        raise FileNotFoundError(f"{RGB_dir_path} not found")
    if not os.path.exists(img_dir_path):
        raise FileNotFoundError(f"{img_dir_path} not found")

    # get the dates in both directories
    common_dates = matching_datetimes_files(RGB_dir_path, img_dir_path)
    # get the full paths to the dates that exist in each directory
    matching_files_RGB_dir, matching_files_img_dir = get_full_paths(
        RGB_dir_path, img_dir_path, common_dates
    )
    # the order must be RGB dir then not RGB dir for other code to work
    # matching_files = sorted(matching_files_RGB_dir) + sorted(matching_files_img_dir)
    files = []
    files.append(sorted(matching_files_RGB_dir))
    files.append(sorted(matching_files_img_dir))
    # creates matrix:  matrix: RGB x number of samples in img_dir_path
    matching_files = np.vstack(files).T
    return matching_files


def RGB_to_infrared(
    RGB_path: str, infrared_path: str, output_path: str, output_type: str
) -> None:
    """Converts two directories of RGB and (NIR/SWIR) imagery to (NDWI/MNDWI) imagery in a directory named
     'NDWI' created at output_path.
     imagery saved as jpg

     to generate NDWI imagery set infrared_path to full path of NIR images
     to generate MNDWI imagery set infrared_path to full path of SWIR images

    Args:
        RGB_path (str): full path to directory containing RGB images
        infrared_path (str): full path to directory containing NIR or SWIR images
        output_path (str): full path to directory to create NDWI/MNDWI directory in
        output_type (str): 'MNDWI' or 'NDWI'
    Based on code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    if output_type.upper() not in ["MNDWI", "NDWI"]:
        logger.error(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
        raise Exception(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
    # matrix:bands(RGB) x number of samples(NIR)
    files = get_files(RGB_path, infrared_path)
    # output_path: directory to store MNDWI or NDWI outputs
    output_path = os.path.join(output_path, output_type.upper())

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        # Read green band from RGB image and cast to float
        green_band = skimage.io.imread(file[0])[:, :, 1].astype("float")
        # Read infrared(SWIR or NIR) and cast to float
        infrared = skimage.io.imread(file[1]).astype("float")
        # Transform 0 to np.NAN
        green_band[green_band == 0] = np.nan
        infrared[infrared == 0] = np.nan
        # Mask out NaNs
        green_band = np.ma.filled(green_band)
        infrared = np.ma.filled(infrared)

        # ensure both matrices have equivalent size
        if not np.shape(green_band) == np.shape(infrared):
            gx, gy = np.shape(green_band)
            nx, ny = np.shape(infrared)
            # resize both matrices to have equivalent size
            green_band = common.scale(
                green_band, np.maximum(gx, nx), np.maximum(gy, ny)
            )
            infrared = common.scale(infrared, np.maximum(gx, nx), np.maximum(gy, ny))

        # output_img(MNDWI/NDWI) imagery formula (Green - SWIR) / (Green + SWIR)
        output_img = np.divide(infrared - green_band, infrared + green_band)
        # Convert the NaNs to -1
        output_img[np.isnan(output_img)] = -1
        # Rescale to be between 0 - 255
        output_img = common.rescale_array(output_img, 0, 255)
        # create new filenames by replacing image type(SWIR/NIR) with output_type
        if output_type.upper() == "MNDWI":
            new_filename = file[1].split(os.sep)[-1].replace("SWIR", output_type)
        if output_type.upper() == "NDWI":
            new_filename = file[1].split(os.sep)[-1].replace("NIR", output_type)

        # save output_img(MNDWI/NDWI) as .jpg in output directory
        skimage.io.imsave(
            output_path + os.sep + new_filename,
            output_img.astype("uint8"),
            check_contrast=False,
            quality=100,
        )

    return output_path


async def fetch(session, url: str, save_path: str):
    model_name = url.split("/")[-1]
    # chunk_size: int = 128
    chunk_size: int = 2048
    async with session.get(url, raise_for_status=True) as r:
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with tqdm.auto.tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {model_name}",
                    initial=0,
                    ascii=False,
                    position=0,
                ) as pbar:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))
        else:
            with open(save_path, "wb") as fd:
                async for chunk in r.content.iter_chunked(chunk_size):
                    fd.write(chunk)


async def fetch_all(session, url_dict):
    tasks = []
    for save_path, url in url_dict.items():
        task = asyncio.create_task(fetch(session, url, save_path))
        tasks.append(task)
    await tqdm.asyncio.tqdm.gather(*tasks)


async def async_download_urls(url_dict: dict) -> None:
    async with aiohttp.ClientSession() as session:
        await fetch_all(session, url_dict)


def run_async_download(url_dict: dict):
    logger.info("run_async_download")
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logger.info("Scheduling task")
    # apply a nested loop to jupyter's event loop for async downloading
    nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_download_urls(url_dict))
    logger.info("Scheduled task")
    logger.info(f"result: {result}")


def get_GPU(num_GPU: str) -> None:
    num_GPU = str(num_GPU)
    if num_GPU == "0":
        logger.info("Not using GPU")
        print("Not using GPU")
        # use CPU (not recommended):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif num_GPU == "1":
        print("Using single GPU")
        logger.info(f"Using 1 GPU")
        # use first available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if int(num_GPU) == 1:
        # read physical GPUs from machine
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        print(f"physical_devices (GPUs):{physical_devices}")
        logger.info(f"physical_devices (GPUs):{physical_devices}")
        if physical_devices:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(physical_devices, "GPU")
            except RuntimeError as e:
                # Visible devices must be set at program startup
                logger.error(e)
                print(e)
        # set mixed precision
        mixed_precision.set_global_policy("mixed_float16")
        # disable memory growth on all GPUs
        for i in physical_devices:
            tf.config.experimental.set_memory_growth(i, True)
            print(f"visible_devices: {tf.config.get_visible_devices()}")
            logger.info(f"visible_devices: {tf.config.get_visible_devices()}")
        # if multiple GPUs are used use mirror strategy
        if int(num_GPU) > 1:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy(
                [p.name.split("/physical_device:")[-1] for p in physical_devices],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(),
            )
            print(f"Number of distributed devices: {strategy.num_replicas_in_sync}")
            logger.info(
                f"Number of distributed devices: {strategy.num_replicas_in_sync}"
            )


def get_url_dict_to_download(models_json_dict: dict) -> dict:
    """Returns dictionary of paths to save files to download
    and urls to download file

    ex.
    {'C:\Home\Project\file.json':"https://website/file.json"}

    Args:
        models_json_dict (dict): full path to files and links

    Returns:
        dict: full path to files and links
    """
    url_dict = {}
    for save_path, link in models_json_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path] = link
        json_filepath = save_path.replace("_fullmodel.h5", ".json")
        if not os.path.isfile(json_filepath):
            json_link = link.replace("_fullmodel.h5", ".json")
            url_dict[json_filepath] = json_link

    return url_dict


def get_sorted_files_with_extension(
    sample_direc: str, file_extensions: List[str]
) -> List[str]:
    """
    Get a sorted list of paths to files that have one of the file_extensions.
    It will return the first set of files that matches the first file_extension, so put the
    file_extension list in order of priority

    Args:
        sample_direc: A string representing the directory path to search for images.
        file_extensions: A list of file extensions to search for.

    Returns:
        A list of file paths for sample images found in the directory.

    """
    sample_filenames = []
    for ext in file_extensions:
        filenames = sorted(tf.io.gfile.glob(os.path.join(sample_direc, f"*{ext}")))
        sample_filenames.extend(filenames)
        if sample_filenames:
            break
    return sample_filenames


class Zoo_Model:
    def __init__(self):
        gdal.UseExceptions()
        self.weights_directory = None
        self.model_types = []
        self.model_list = []
        self.metadata_dict = {}
        self.settings = {}
        # create default settings
        self.set_settings()

    def clear_zoo_model(self):
        self.weights_directory = None
        self.model_types = []
        self.model_list = []
        self.metadata_dict = {}
        self.settings = {}

    def set_settings(self, **kwargs):
        """
        Saves the settings for downloading data by updating the `self.settings` dictionary with the provided key-value pairs.
        If any of the keys are missing, they will be set to their default value as specified in `default_settings`.

        Example: set_settings(sat_list=sat_list, dates=dates,**more_settings)

        Args:
        **kwargs: Keyword arguments representing the key-value pairs to be added to or updated in `self.settings`.

        Returns:
        None
        """
        # Check if any of the keys are missing
        # if any keys are missing set the default value
        default_settings = {
            "sample_direc": None,
            "use_GPU": "0",
            "implementation": "BEST",
            "model_type": "segformer_RGB_4class_8190958",
            "otsu": False,
            "tta": False,
            "cloud_thresh": 0.5,  # threshold on maximum cloud cover
            "dist_clouds": 300,  # ditance around clouds where shoreline can't be mapped
            "output_epsg": 4326,  # epsg code of spatial reference system desired for the output
            "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
            # minimum area (in metres^2) for an object to be labelled as a beach
            "min_beach_area": 4500,
            # minimum length (in metres) of shoreline perimeter to be valid
            "min_length_sl": 100,
            # switch this parameter to True if sand pixels are masked (in black) on many images
            "cloud_mask_issue": False,
            # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
            "sand_color": "default",
            "pan_off": "False",  # if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
            "max_dist_ref": 25,
            "along_dist": 25,  # along-shore distance to use for computing the intersection
            "min_points": 3,  # minimum number of shoreline points to calculate an intersection
            "max_std": 15,  # max std for points around transect
            "max_range": 30,  # max range for points around transect
            "min_chainage": -100,  # largest negative value along transect (landwards of transect origin)
            "multiple_inter": "auto",  # mode for removing outliers ('auto', 'nan', 'max')
            "prc_multiple": 0.1,  # percentage of the time that multiple intersects are present to use the max
            "percent_no_data": 50.0,  # percentage of no data pixels allowed in an image (doesn't work for npz)
            "model_session_path": "",  # path to model session file
            "apply_cloud_mask": True,  # whether to apply cloud mask to images or not
            "drop_intersection_pts": False, # whether to drop intersection points not on the transect
        }
        if kwargs:
            self.settings.update({key: value for key, value in kwargs.items()})
    
        for key, value in default_settings.items():
            self.settings.setdefault(key, value)
            
        logger.info(f"Settings: {self.settings}")
        return self.settings.copy()

    def get_settings(self):
        SETTINGS_NOT_FOUND = (
            "No settings found. Click save settings or load a config file."
        )
        logger.info(f"self.settings: {self.settings}")
        if self.settings is None or self.settings == {}:
            raise Exception(SETTINGS_NOT_FOUND)
        return self.settings

    def preprocess_data(
        self, src_directory: str, model_dict: dict, img_type: str
    ) -> dict:
        """
        Preprocesses the data in the source directory and updates the model dictionary with the processed data.

        Args:
            src_directory (str): The path to the source directory containing the ROI's data
            model_dict (dict): The dictionary containing the model configuration and parameters.
            img_type (str): The type of imagery to generate. Must be one of "RGB", "NDWI", "MNDWI", or "RGB+MNDWI+NDWI".

        Returns:
            dict: The updated model dictionary containing the paths to the processed data.
        """
        # if configs do not exist then raise an error and do not save the session
        if not file_utilities.validate_config_files_exist(src_directory):
            logger.warning(
                f"Config files config.json or config_gdf.geojson do not exist in roi directory { src_directory}\n This means that the download did not complete successfully."
            )
            raise FileNotFoundError(
                f"Config files config.json or config_gdf.geojson do not exist in roi directory { src_directory}\n This means that the download did not complete successfully."
            )
        # get full path to directory named 'RGB' containing RGBs
        RGB_path = file_utilities.find_directory_recursively(src_directory, name="RGB")
        # convert RGB to MNDWI, NDWI,or 5 band
        model_dict["sample_direc"] = get_imagery_directory(img_type, RGB_path)
        return model_dict

    def run_model_and_extract_shorelines(self,
                                         input_directory:str,
                                         session_name:str,
                                         shoreline_path:str="",
                                         transects_path:str="",
                                         shoreline_extraction_area_path:str="",
                                         use_tta:bool =False,
                                         use_otsu:bool = False,
                                         percent_no_data:float = 50.0,
                                         use_GPU:str="0",
                                         ):
        """
        Runs the model and extracts shorelines using the segmented imagery.

        Args:
            input_directory (str): The directory containing the input images.
            session_name (str): The name of the session.
            shoreline_settings (dict): A dictionary containing shoreline extraction settings.
            img_type (str): The type of input images.
            model_implementation (str): The implementation of the model.
            model_name (str): The name of the model.
            shoreline_path (str, optional): The path to save the extracted shorelines. Defaults to "".
            transects_path (str, optional): The path to save the extracted transects. Defaults to "".
            shoreline_extraction_area_path (str, optional): The path to the shoreline extraction area. Defaults to "".
            use_tta (bool, optional): Whether to use test-time augmentation. Defaults to False.
            use_otsu (bool, optional): Whether to use Otsu thresholding. Defaults to False.
            percent_no_data (float, optional): The percentage of no-data pixels in the input images. Defaults to 50.0.
            use_GPU (str, optional): The GPU device to use. Defaults to "0".
        """
        settings = self.get_settings()
        model_name = settings.get('model_type', None)
        if model_name is None:
            raise ValueError("Please select a model type.")
        
        img_type = settings.get('img_type', None)
        if img_type is None:
            raise ValueError("Please select an input image type.")
        model_implementation = settings.get('implementation', "BEST")
        use_GPU = settings.get('use_GPU', "0")
        use_otsu = settings.get('otsu', False)
        use_tta = settings.get('tta', False)
        percent_no_data = settings.get('percent_no_data', 0.5)
        
        # make a progress bar to show the progress of the model and shoreline extraction
        prog_bar = tqdm.auto.tqdm(range(2),
            desc=f"Running {model_name} model and extracting shorelines",
            leave=True,
        )
        # run the model
        self.run_model(
            img_type,
            model_implementation,
            session_name,
            input_directory,
            model_name=model_name,
            use_GPU=use_GPU,
            use_otsu=use_otsu,
            use_tta=use_tta,
            percent_no_data=percent_no_data,
        )
        prog_bar.update(1)
        prog_bar.set_description_str(
                                desc=f"Ran model now extracting shorelines", refresh=True
        )
        sessions_path = os.path.join(core_utilities.get_base_dir(), "sessions")
        session_directory = file_utilities.create_directory(sessions_path, session_name)
        # extract shorelines using the segmented imagery
        self.extract_shorelines_with_unet(
            settings,
            session_directory,
            session_name,
            shoreline_path,
            transects_path,
            shoreline_extraction_area_path,
        )
        prog_bar.update(1)
        prog_bar.set_description_str(
                                desc=f"Finished running model and extracting shorelines", refresh=True
        )


    def extract_shorelines_with_unet(
        self,
        settings: dict,
        session_path: str,
        session_name: str,
        shoreline_path: str = "",
        transects_path: str = "",
        shoreline_extraction_area_path: str = "",
        **kwargs: dict,
    ) -> None:
        """
        Extracts shorelines using the U-Net model.

        Args:
            settings (dict): A dictionary containing the settings for shoreline extraction.
            session_path (str): The path to the model session directory containing the model outputs and configuration files.
            session_name (str): The name of the session to save the extracted shorelines to.
            shoreline_path (str, optional): The path to the shoreline data. Defaults to "".
            - If a geojson file is not provided, the program will attempt to load default shorelines and if that fails it will raise an error.
            transects_path (str, optional): The path to the transects data. Defaults to "".
            - If a geojson file is not provided, the program will attempt to load default transects and if that fails it will raise an error.
            **kwargs (dict): Additional keyword arguments.
            shoreline_extraction_area_path (str, optional): The path to the shoreline extraction area. Defaults to "".
        Returns:
            None
        """
        logger.info(f"extract_shoreline_settings: {settings}")

        # save the selected model session
        settings["model_session_path"] = session_path
        self.set_settings(**settings)
        settings = self.get_settings()

        # create session path to store extracted shorelines and transects
        base_path = core_utilities.get_base_dir()
        new_session_path = base_path / 'sessions' / session_name

        
        new_session_path.mkdir(parents=True, exist_ok=True)

        # load the ROI settings from the config file
        try:
            config = file_utilities.load_json_data_from_file(
                session_path, "config.json"
            )
            # get the roi_id from the config file @todo this won't work for multiple ROIs
            if config.get("roi_ids"):
                roi_id = config["roi_ids"][0]
            roi_settings = {roi_id:config[roi_id]}
        except (KeyError, ValueError) as e:
            logger.error(f"{roi_id} ROI settings did not exist: {e}")
            if roi_id is None:
                logger.error(f"roi_id was None config: {config}")
                raise Exception(f"The session loaded was \n config: {config}")
            else:
                logger.error(
                    f"roi_id {roi_id} existed but not found in config: {config}"
                )
                raise Exception(
                    f"The roi ID {roi_id} did not exist is the config.json \n config.json: {config}"
                )
        logger.info(f"roi_settings: {roi_settings}")

        # read ROI from config geojson file
        config_geojson_location = file_utilities.find_file_recursively(
            session_path, "config_gdf.geojson"
        )
        logger.info(f"config_geojson_location: {config_geojson_location}")
        config_gdf = geodata_processing.read_gpd_file(config_geojson_location)
        roi_gdf = config_gdf[config_gdf["id"] == roi_id]
        if roi_gdf.empty:
            logger.error(
                f"{roi_id} ROI ID did not exist in geodataframe: {config_geojson_location}"
            )
            raise ValueError(
                f"{roi_id} ROI ID did not exist in geodataframe: {config_geojson_location}"
            )

        # get roi_id from source directory path in model settings
        model_settings = file_utilities.load_json_data_from_file(
            session_path, "model_settings.json"
        )

        # save model settings to session path
        model_settings_path = os.path.join(new_session_path, "model_settings.json")
        file_utilities.to_file(model_settings, model_settings_path)

        # load transects and shorelines
        output_epsg = settings["output_epsg"]
        transects_gdf = geodata_processing.create_geofeature_geodataframe(
            transects_path, roi_gdf, output_epsg, "transect"
        )
        shoreline_gdf = geodata_processing.create_geofeature_geodataframe(
            shoreline_path, roi_gdf, output_epsg, "shoreline"
        )
        shoreline_extraction_area_gdf = None
        # load the shoreline extraction area from the geojson file
        if os.path.exists(shoreline_extraction_area_path):
            shoreline_extraction_area_gdf = geodata_processing.load_feature_from_file(shoreline_extraction_area_path, "shoreline_extraction_area")

        # Update the CRS to the most accurate crs for the ROI this makes extracted shoreline more accurate
        new_espg = common.get_most_accurate_epsg(output_epsg, roi_gdf)
        settings["output_epsg"] = new_espg
        self.set_settings(output_epsg=new_espg)
        # convert the ROI to the new CRS
        roi_gdf = roi_gdf.to_crs(output_epsg)

        # save the config files to the new session location
        common.save_config_files(
            new_session_path,
            roi_ids=[roi_id],
            roi_settings=roi_settings,
            shoreline_settings=settings,
            transects_gdf=transects_gdf,
            shorelines_gdf=shoreline_gdf,
            roi_gdf=roi_gdf,
            epsg_code="epsg:4326",
            shoreline_extraction_area_gdf = shoreline_extraction_area_gdf,
        )

        # extract shorelines
        extracted_shorelines = extracted_shoreline.Extracted_Shoreline()
        extracted_shorelines = (
            extracted_shorelines.create_extracted_shorelines_from_session(
                roi_id,
                shoreline_gdf,
                roi_settings[roi_id],
                settings,
                session_path,
                new_session_path,
                shoreline_extraction_area=shoreline_extraction_area_gdf,
                **kwargs,
            )
        )

        # save extracted shorelines, detection jpgs, configs, model settings files to the session directory
        common.save_extracted_shorelines(extracted_shorelines, new_session_path)

        # common.save_extracted_shoreline_figures(extracted_shorelines, new_session_path)
        print(f"Saved extracted shorelines to {new_session_path}")

        # transects must be in the same CRS as the extracted shorelines otherwise intersections will all be NAN
        if not transects_gdf.empty:
            transects_gdf = transects_gdf.to_crs(new_espg)

        # compute intersection between extracted shorelines and transects
        cross_distance_transects = extracted_shoreline.compute_transects_from_roi(
            extracted_shorelines.dictionary, transects_gdf, settings
        )

        first_key = next(iter(cross_distance_transects))
        logger.info(
            f"cross_distance_transects.keys(): {cross_distance_transects.keys()}"
        )
        logger.info(
            f"Sample of transect intersections for first key: {list(islice(cross_distance_transects[first_key], 3))}"
        )

        # save transect shoreline intersections to csv file if they exist
        if cross_distance_transects == 0:
            logger.warning("No transect shoreline intersections.")
            print("No transect shoreline intersections.")
        else:
            transect_settings = self.get_settings()
            transect_settings["output_epsg"] = new_espg
            drop_intersection_pts=self.get_settings().get('drop_intersection_pts', False)
            common.save_transects(
                new_session_path,
                cross_distance_transects,
                extracted_shorelines.dictionary,
                transect_settings,
                transects_gdf,
                drop_intersection_pts
            )

    def postprocess_data(
        self, preprocessed_data: dict, session: sessions.Session, roi_directory: str
    ):
        """Moves the model outputs from
        as well copies the config files from the roi directory to the session directory

        Args:
            preprocessed_data (dict): dictionary of inputs to the model
            session (sessions.Session): session object that's used to keep track of session
            saves the session to the sessions directory
            roi_directory (str):  directory in data that contains downloaded data for a single ROI
            typically starts with "ID_{roi_id}"
        """
        # get roi_ids
        session_path = session.path
        outputs_path = os.path.join(preprocessed_data["sample_direc"], "out")
        if not os.path.exists(outputs_path):
            logger.warning(f"No model outputs were generated")
            print(f"No model outputs were generated")
            raise Exception(f"No model outputs were generated. Check if {roi_directory} contained enough data to run the model or try raising the percentage of no data allowed.")
        logger.info(f"Moving from {outputs_path} files to {session_path}")

        # if configs do not exist then raise an error and do not save the session
        if not file_utilities.validate_config_files_exist(roi_directory):
            logger.warning(
                f"Config files config.json or config_gdf.geojson do not exist in roi directory {roi_directory}"
            )
            raise FileNotFoundError(
                f"Config files config.json or config_gdf.geojson do not exist in roi directory {roi_directory}"
            )
        # modify the config.json to only have the ROI ID that was used and save to session directory
        roi_id = file_utilities.extract_roi_id(roi_directory)
        common.save_new_config(
            os.path.join(roi_directory, "config.json"),
            roi_id,
            os.path.join(session_path, "config.json"),
        )
        # Copy over the config_gdf.geojson file
        config_gdf_path = os.path.join(roi_directory, "config_gdf.geojson")
        if os.path.exists(config_gdf_path):
            # Read in the GeoJSON file using geopandas
            gdf = gpd.read_file(config_gdf_path)

            # Project the GeoDataFrame to EPSG:4326
            gdf_4326 = gdf.to_crs("EPSG:4326")

            # Save the projected GeoDataFrame to a new GeoJSON file
            gdf_4326.to_file(
                os.path.join(session_path, "config_gdf.geojson"), driver="GeoJSON"
            )
        model_settings_path = os.path.join(session_path, "model_settings.json")
        file_utilities.write_to_json(model_settings_path, preprocessed_data)

        # copy files from out to session folder
        file_utilities.move_files(outputs_path, session_path, delete_src=True)
        session.save(session.path)

    def prepare_model(self, model_implementation: str, model_id: str):
        """
        Prepares the model for use by downloading the required files and loading the model.

        Args:
            model_implementation (str): The model implementation either 'BEST' or 'ENSEMBLE'
            model_id (str): The ID of the model.
        """
        self.clear_zoo_model()
        # create the model directory
        self.weights_directory = self.get_model_directory(model_id)
        logger.info(f"self.weights_directory:{self.weights_directory}")

        self.download_model(model_implementation, model_id, self.weights_directory)
        weights_list = self.get_weights_list(model_implementation)

        # Load the model from the config files
        model, model_list, config_files, model_types = self.get_model(weights_list)
        logger.info(f"self.TARGET_SIZE: {self.TARGET_SIZE}")
        logger.info(f"self.N_DATA_BANDS: {self.N_DATA_BANDS}")
        logger.info(f"self.TARGET_SIZE: {self.TARGET_SIZE}")

        self.model_types = model_types
        self.model_list = model_list
        self.metadata_dict = self.get_metadatadict(
            weights_list, config_files, model_types
        )
        logger.info(f"self.metadatadict: {self.metadata_dict}")

    def get_metadatadict(
            self, weights_list: list, config_files: list, model_types: list
        ) -> dict:
            """
            Returns a dictionary containing metadata information.

            Args:
                weights_list (list): A list of model weights.
                config_files (list): A list of configuration files.
                model_types (list): A list of model types.

            Returns:
                dict: A dictionary containing the metadata information.
            """
            metadatadict = {}
            metadatadict["model_weights"] = weights_list
            metadatadict["config_files"] = config_files
            metadatadict["model_types"] = model_types
            return metadatadict

    def get_classes(self, model_directory_path: str):
            """
            Retrieves the classes from the specified model directory.

            Args:
                model_directory_path (str): The path to the model directory.

            Returns:
                list: A list of classes.
            """
            class_path = os.path.join(model_directory_path, "classes.txt")
            classes = common.read_text_file(class_path)
            return classes

    def run_model(
            self,
            img_type: str,
            model_implementation: str,
            session_name: str,
            src_directory: str,
            model_name: str,
            use_GPU: str,
            use_otsu: bool,
            use_tta: bool,
            percent_no_data: float,
        ):
            """
            Runs the model for image segmentation.

            Args:
                img_type (str): The type of image.
                model_implementation (str): The implementation of the model.
                session_name (str): The name of the session.
                src_directory (str): The directory of RGB images.
                model_name (str): The name of the model.
                use_GPU (str): Whether to use GPU or not.
                use_otsu (bool): Whether to use Otsu thresholding or not.
                use_tta (bool): Whether to use test-time augmentation or not.
                percent_no_data (float): The percentage of no data.

            Returns:
                None
            """
            logger.info(f"Selected directory of RGBs: {src_directory}")
            logger.info(f"session name: {session_name}")
            logger.info(f"model_name: {model_name}")
            logger.info(f"model_implementation: {model_implementation}")
            logger.info(f"use_GPU: {use_GPU}")
            logger.info(f"use_otsu: {use_otsu}")
            logger.info(f"use_tta: {use_tta}")

            print(f"Running model {model_name}")
            self.prepare_model(model_implementation, model_name)

            # create a session
            session = sessions.Session()
            sessions_path = file_utilities.create_directory(core_utilities.get_base_dir(), "sessions")
            session_path = file_utilities.create_directory(sessions_path, session_name)

            session.path = session_path
            session.name = session_name
            model_dict = {
                "use_GPU": use_GPU,
                "sample_direc": "",
                "implementation": model_implementation,
                "model_type": model_name,
                "otsu": use_otsu,
                "tta": use_tta,
                "percent_no_data": percent_no_data,
            }
            # get parent roi_directory from the selected imagery directory
            roi_directory = file_utilities.find_parent_directory(
                src_directory, "ID_", "data"
            )

            print(f"Preprocessing the data at {roi_directory}")
            model_dict = self.preprocess_data(roi_directory, model_dict, img_type)
            logger.info(f"model_dict: {model_dict}")

            self.compute_segmentation(model_dict, percent_no_data)
            self.postprocess_data(model_dict, session, roi_directory)
            session.add_roi_ids([file_utilities.extract_roi_id(roi_directory)])
            print(f"\n Model results saved to {session.path}")

    def get_model_directory(self, model_id: str):
        # Create a directory to hold the downloaded models
        downloaded_models_path = common.get_downloaded_models_dir()
        model_directory = file_utilities.create_directory(
            downloaded_models_path, model_id
        )
        return model_directory

    def get_files_for_seg(
        self,
        sample_direc: str,
        avoid_patterns: List[str] = [],
        percent_no_data: float = 0.50,
    ) -> list:
        """
        Returns a list of files to be segmented.

        The function reads in the image filenames as either (`.npz`) OR (`.jpg`, or `.png`)
        and returns a sorted list of the file paths.

        Args:
        - sample_direc (str): The directory containing files to be segmented.
        - avoid_patterns (List[str], optional): A list of file names to be avoided.Don't include any file extensions. Default is [].
        - percent_no_data (float, optional): The percentage of no data pixels allowed in an image. Default is 0.50.
        Returns:
        - list: A list of full pathes files to be segmented.
        """
        logger.info(f"Searching directory for files: {sample_direc}")
        file_extensions = [".npz", ".jpg", ".png"]
        model_ready_files = get_sorted_files_with_extension(
            sample_direc, file_extensions
        )
        # filter out files whose filenames match any of the avoid_patterns
        model_ready_files = file_utilities.filter_files(
            model_ready_files, avoid_patterns
        )
        # filter out files with no data pixels greater than percent_no_data
        len_before = len(model_ready_files)
        model_ready_files = filter_no_data_pixels(model_ready_files, percent_no_data)
        print(f"From {len_before} files {len_before - len(model_ready_files)} files were filtered out due to no data pixels percentage being greater than {percent_no_data*100}%.")
        
        return model_ready_files

    def compute_segmentation(
            self,
            preprocessed_data: dict,
            percent_no_data: float = 0.50,
        ):
            """
            Compute the segmentation for a given set of preprocessed data.

            Args:
                preprocessed_data (dict): A dictionary containing preprocessed data.
                    This dictionary should contain the following keys:
                    - sample_direc (str): The directory containing the sample images.
                    - tta (bool): Whether to use test-time augmentation.
                    - otsu (bool): Whether to use Otsu thresholding.
                    
                percent_no_data (float, optional): The max ercentage of no data pixels allowed in the image. Defaults to 0.50.

            Returns:
                None
            """
            sample_direc = preprocessed_data["sample_direc"]
            use_tta = preprocessed_data["tta"]
            use_otsu = preprocessed_data["otsu"]
            # Create list of files of types .npz,.jpg, or .png to run model on
            files_to_segment = self.get_files_for_seg(
                sample_direc, avoid_patterns=[], percent_no_data=percent_no_data
            )
            logger.info(f"files_to_segment: {files_to_segment}")
            if self.model_types[0] != "segformer":
                ### mixed precision
                from tensorflow.keras import mixed_precision

                mixed_precision.set_global_policy("mixed_float16")
            # Compute the segmentation for each of the files
            print(f"Found {len(files_to_segment)} files to run on model on")
            for file_to_seg in tqdm.auto.tqdm(files_to_segment, desc="Applying Model"):
                do_seg(
                    file_to_seg,
                    self.model_list,
                    self.metadata_dict,
                    self.model_types[0],
                    sample_direc=sample_direc,
                    NCLASSES=self.NCLASSES,
                    N_DATA_BANDS=self.N_DATA_BANDS,
                    TARGET_SIZE=self.TARGET_SIZE,
                    TESTTIMEAUG=use_tta,
                    WRITE_MODELMETADATA=False,
                    OTSU_THRESHOLD=use_otsu,
                    profile="meta",
                )

    def get_model(self, weights_list: list):
        model_list = []
        config_files = []
        model_types = []
        logger.info(f"weights_list: {weights_list}")
        if weights_list == []:
            raise Exception("No Model Info Passed")
        for weights in weights_list:
            # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
            # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
            weights = weights.strip()
            config_file = weights.replace(".h5", ".json").replace("weights", "config")
            if "fullmodel" in config_file:
                config_file = config_file.replace("_fullmodel", "")
            with open(config_file) as f:
                config = json.load(f)
            self.TARGET_SIZE = config.get("TARGET_SIZE")
            MODEL = config.get("MODEL")
            self.NCLASSES = config.get("NCLASSES")
            KERNEL = config.get("KERNEL")
            STRIDE = config.get("STRIDE")
            FILTERS = config.get("FILTERS")
            self.N_DATA_BANDS = config.get("N_DATA_BANDS")
            DROPOUT = config.get("DROPOUT")
            DROPOUT_CHANGE_PER_LAYER = config.get("DROPOUT_CHANGE_PER_LAYER")
            DROPOUT_TYPE = config.get("DROPOUT_TYPE")
            USE_DROPOUT_ON_UPSAMPLING = config.get("USE_DROPOUT_ON_UPSAMPLING")
            DO_TRAIN = config.get("DO_TRAIN")
            LOSS = config.get("LOSS")
            PATIENCE = config.get("PATIENCE")
            MAX_EPOCHS = config.get("MAX_EPOCHS")
            VALIDATION_SPLIT = config.get("VALIDATION_SPLIT")
            RAMPUP_EPOCHS = config.get("RAMPUP_EPOCHS")
            SUSTAIN_EPOCHS = config.get("SUSTAIN_EPOCHS")
            EXP_DECAY = config.get("EXP_DECAY")
            START_LR = config.get("START_LR")
            MIN_LR = config.get("MIN_LR")
            MAX_LR = config.get("MAX_LR")
            FILTER_VALUE = config.get("FILTER_VALUE")
            DOPLOT = config.get("DOPLOT")
            ROOT_STRING = config.get("ROOT_STRING")
            USEMASK = config.get("USEMASK")
            AUG_ROT = config.get("AUG_ROT")
            AUG_ZOOM = config.get("AUG_ZOOM")
            AUG_WIDTHSHIFT = config.get("AUG_WIDTHSHIFT")
            AUG_HEIGHTSHIFT = config.get("AUG_HEIGHTSHIFT")
            AUG_HFLIP = config.get("AUG_HFLIP")
            AUG_VFLIP = config.get("AUG_VFLIP")
            AUG_LOOPS = config.get("AUG_LOOPS")
            AUG_COPIES = config.get("AUG_COPIES")
            REMAP_CLASSES = config.get("REMAP_CLASSES")

            try:
                model = tf.keras.models.load_model(weights)
                #  nclasses=NCLASSES, may have to replace nclasses with NCLASSES
            except BaseException:
                if MODEL == "resunet":
                    model = custom_resunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )
                elif MODEL == "unet":
                    model = custom_unet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )

                elif MODEL == "simple_resunet":
                    # num_filters = 8 # initial filters
                    model = simple_resunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                # 346,564
                elif MODEL == "simple_unet":
                    model = simple_unet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL == "satunet":
                    model = simple_satunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=self.NCLASSES,  # [NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        filters=FILTERS,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL == "segformer":
                    id2label = {}
                    for k in range(self.NCLASSES):
                        id2label[k] = str(k)
                    model = segformer(id2label, num_classes=self.NCLASSES)
                    model.compile(optimizer="adam")
                # 242,812
                else:
                    raise Exception(
                        f"An unknown model type {MODEL} was received. Please select a valid model.\n \
                        Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'"
                    )

                # Load in the custom loss function from doodleverse_utils
                model.compile(
                    optimizer="adam", loss=dice_coef_loss(self.NCLASSES)
                )  # , metrics = [iou_multi(self.NCLASSESNCLASSES), dice_multi(self.NCLASSESNCLASSES)])

                model.load_weights(weights)

            self.model_types.append(MODEL)
            self.model_list.append(model)
            config_files.append(config_file)
        return model, self.model_list, config_files, self.model_types

    def get_metadatadict(
        self, weights_list: list, config_files: list, model_types: list
    ) -> dict:
        """Returns a dictionary containing metadata about the models.

        Args:
            weights_list (list): A list of model weights.
            config_files (list): A list of model configuration files.
            model_types (list): A list of model types.

        Returns:
            dict: A dictionary containing metadata about the models. The keys
            are 'model_weights', 'config_files', and 'model_types', and the
            values are the corresponding input lists.

        Example:
            weights = ['weights1.h5', 'weights2.h5']
            configs = ['config1.json', 'config2.json']
            types = ['unet', 'resunet']
            metadata = get_metadatadict(weights, configs, types)
            print(metadata)
            # Output: {'model_weights': ['weights1.h5', 'weights2.h5'],
            #          'config_files': ['config1.json', 'config2.json'],
            #          'model_types': ['unet', 'resunet']}
        """
        metadatadict = {}
        metadatadict["model_weights"] = weights_list
        metadatadict["config_files"] = config_files
        metadatadict["model_types"] = model_types
        return metadatadict

    def get_weights_list(self, model_choice: str = "ENSEMBLE") -> List[str]:
        """Returns a list of the model weights files (.h5) within the weights directory.

        Args:
            model_choice (str, optional): The type of model weights to return.
                Valid choices are 'ENSEMBLE' (default) to return all available
                weights files or 'BEST' to return only the best model weights file.

        Returns:
            list: A list of strings representing the file paths to the model weights
            files in the weights directory.

        Raises:
            FileNotFoundError: If the BEST_MODEL.txt file is not found in the weights directory.

        Example:
            trainer = ModelTrainer(weights_direc='/path/to/weights')
            weights_list = trainer.get_weights_list(model_choice='ENSEMBLE')
            print(weights_list)
            # Output: ['/path/to/weights/model1.h5', '/path/to/weights/model2.h5', ...]

            best_weights_list = trainer.get_weights_list(model_choice='BEST')
            print(best_weights_list)
            # Output: ['/path/to/weights/best_model.h5']
        """
        logger.info(f"{model_choice}")
        if model_choice == "ENSEMBLE":
            weights_list = glob(os.path.join(self.weights_directory, "*.h5"))
            logger.info(f"weights_list: {weights_list}")
            logger.info(f"{len(weights_list)} sets of model weights were found ")
            return weights_list
        elif model_choice == "BEST":
            # read model name (fullmodel.h5) from BEST_MODEL.txt
            with open(os.path.join(self.weights_directory, "BEST_MODEL.txt")) as f:
                model_name = f.readline()
            logger.info(f"model_name: {model_name}")
            # remove any leading or trailing whitespace and newline characters
            model_name = model_name.strip()
            weights_list = [os.path.join(self.weights_directory, model_name)]
            logger.info(f"weights_list: {weights_list}")
            logger.info(f"{len(weights_list)} sets of model weights were found ")
            return weights_list
        else:
            raise ValueError(
                f"Invalid model_choice: {model_choice}. Valid choices are 'ENSEMBLE' or 'BEST'."
            )

    def download_best(
        self, available_files: List[dict], model_path: str, model_id: str
    ):
        """
        Downloads the best model file and its corresponding JSON and classes.txt files from the given list of available files.

        Args:
            available_files (list): A list of files available to download.
            model_path (str): The local directory where the downloaded files will be stored.
            model_id (str): The ID of the model being downloaded.

        Raises:
            ValueError: If BEST_MODEL.txt file is not found in the given model_id.

        Returns:
            None
        """
        download_dict = {}
        # download best_model.txt and read the name of the best model
        best_model_json = next(
            (f for f in available_files if f["key"] == "BEST_MODEL.txt"), None
        )
        if best_model_json is None:
            raise ValueError(f"Cannot find BEST_MODEL.txt in {model_id}")
        # download best model file to check if it exists
        BEST_MODEL_txt_path = os.path.join(model_path, "BEST_MODEL.txt")
        logger.info(f"model_path for BEST_MODEL.txt: {BEST_MODEL_txt_path}")
        # if best BEST_MODEL.txt file not exist then download it
        if not os.path.isfile(BEST_MODEL_txt_path):
            common.download_url(
                best_model_json["links"]["self"],
                BEST_MODEL_txt_path,
                "Downloading best_model.txt",
            )

        with open(BEST_MODEL_txt_path, "r") as f:
            best_model_filename = f.read().strip()
        # get the json data of the best model _fullmodel.h5 file
        best_model_filename = best_model_filename.strip()
        best_json_filename = best_model_filename.replace("_fullmodel.h5", ".json")
        best_modelcard_filename = best_model_filename.replace(
            "_fullmodel.h5", "_modelcard.json"
        ).replace("_segformer", "")

        # download best model files(.h5, .json) file
        download_filenames = [
            best_json_filename,
            best_model_filename,
            best_modelcard_filename,
        ]
        logger.info(f"download_filenames: {download_filenames}")
        download_dict.update(
            get_files_to_download(
                available_files, download_filenames, model_id, model_path
            )
        )
        download_dict = check_if_files_exist(download_dict)
        # download the files that don't exist
        logger.info(f"URLs to download: {download_dict}")
        # if any files are not found locally download them asynchronous
        if download_dict != {}:
            download_status = downloads.download_url_dict(download_dict)
            if download_status == False:
                raise Exception("Download failed")

    def download_ensemble(
        self, available_files: List[dict], model_path: str, model_id: str
    ):
        """
        Downloads all the model files and their corresponding JSON and classes.txt files from the given list of available files, for an ensemble model.

        Args:
            available_files (list): A list of files available to download.
            model_path (str): The local directory where the downloaded files will be stored.
            model_id (str): The ID of the model being downloaded.

        Raises:
            Exception: If no .h5 files or corresponding .json files are found in the given model_id.

        Returns:
            None
        """
        download_dict = {}
        # get json and models
        all_models_reponses = [f for f in available_files if f["key"].endswith(".h5")]
        all_model_names = [f["key"] for f in all_models_reponses]
        json_file_names = [
            model_name.replace("_fullmodel.h5", ".json")
            for model_name in all_model_names
        ]
        modelcard_file_names = [
            model_name.replace("_fullmodel.h5", "_modelcard.json").replace(
                "_segformer", ""
            )
            for model_name in all_model_names
        ]
        all_json_reponses = []

        # for each filename online check if there a .json file
        for available_file in available_files:
            if available_file["key"] in json_file_names + modelcard_file_names:
                all_json_reponses.append(available_file)
        if len(all_models_reponses) == 0:
            raise Exception(f"Cannot find any .h5 files at {model_id}")
        if len(all_json_reponses) == 0:
            raise Exception(
                f"Cannot find corresponding .json or .modelcard.json files for .h5 files at {model_id}"
            )

        logger.info(f"all_models_reponses : {all_models_reponses }")
        logger.info(f"all_json_reponses : {all_json_reponses }")
        for response in all_models_reponses + all_json_reponses:
            # get the link of the best model
            link = response["links"]["self"]
            filename = response["key"]
            filepath = os.path.join(model_path, filename)
            download_dict[filepath] = link

        download_dict.update(
            get_files_to_download(available_files, [], model_id, model_path)
        )
        download_dict = check_if_files_exist(download_dict)
        # download the files that don't exist
        logger.info(f"download_dict: {download_dict}")
        # if any files are not found locally download them asynchronous
        if download_dict != {}:
            download_status = downloads.download_url_dict(download_dict)
            if download_status == False:
                raise Exception("Download failed")

    def download_model(
        self, model_choice: str, model_id: str, model_path: str = None
    ) -> None:
        """downloads model specified by zenodo id in model_id.

        Downloads best model is model_choice = 'BEST' or all models in
        zenodo release if model_choice = 'ENSEMBLE'

        Args:
            model_choice (str): 'BEST' or 'ENSEMBLE'
            model_id (str): name of model followed by underscore zenodo_id ex.'orthoCT_RGB_2class_7574784'
            model_path (str): path to directory to save the downloaded files to
        """
        # Extract the Zenodo ID from the dataset ID
        zenodo_id = model_id.split("_")[-1]
        # get list of files available in zenodo release
        json_content = get_zenodo_release(zenodo_id)
        available_files = json_content["files"]

        # Download the best model if best or all models if ensemble
        if model_choice.upper() == "BEST":
            self.download_best(available_files, model_path, model_id)
        elif model_choice.upper() == "ENSEMBLE":
            self.download_ensemble(available_files, model_path, model_id)
