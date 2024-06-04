import asyncio
import concurrent.futures
from datetime import datetime
import glob
import json
import logging
import math
import os
import platform
import shutil
import zipfile

import aiohttp
import area
import ee
import geopandas as gpd
import nest_asyncio
import tqdm
import tqdm.asyncio
import tqdm.auto
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
from typing import Collection, List, Optional, Tuple, Union

from coastseg import common
from coastseg import file_utilities

logger = logging.getLogger(__name__)


def get_collection_by_tier(
    polygon: List[List[float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    satellite: str,
    tier: int,
    max_cloud_cover: float = 95,
    months_list = None
) -> Union[ee.ImageCollection, None]:
    """
    This function takes the required parameters and returns an ImageCollection from
    the specified satellite and tier filtered by the given polygon, date range, and cloud cover.

    Args:
    polygon (List[List[float]]): The polygon to filter the ImageCollection by.
    start_date (Union[str, datetime]): The start date to filter the ImageCollection by.
    end_date (Union[str, datetime]): The end date to filter the ImageCollection by.
    satellite (str): The satellite to select the ImageCollection from.
    tier (int): The tier of the satellite data.
    max_cloud_cover (float): The maximum cloud cover percentage for the entire scene (not just the roi) to filter the ImageCollection by.

    Returns:
    ee.ImageCollection or None: The filtered ImageCollection or None if the inputs are invalid.
    """
    if months_list is None:
        months_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Converting datetime objects to string if passed as datetime
    if isinstance(start_date, datetime):
        start_date = start_date.isoformat()
    if isinstance(end_date, datetime):
        end_date = end_date.isoformat()

    # Define collection names for tier 1 and tier 2
    col_names = {
        1: {
            "L5": "LANDSAT/LT05/C02/T1_TOA",
            "L7": "LANDSAT/LE07/C02/T1_TOA",
            "L8": "LANDSAT/LC08/C02/T1_TOA",
            "L9": "LANDSAT/LC09/C02/T1_TOA",
            "S2": "COPERNICUS/S2_HARMONIZED",
        },
        2: {
            "L5": "LANDSAT/LT05/C02/T2_TOA",
            "L7": "LANDSAT/LE07/C02/T2_TOA",
            "L8": "LANDSAT/LC08/C02/T2_TOA",
        },
    }

    # Validate inputs and get collection name
    if tier not in col_names:
        print(f"Invalid tier ({tier})")
        return None
    # not all satellites are in tier 2 return None for any that are not
    if satellite not in col_names[tier]:
        return None

    collection_name = col_names[tier][satellite]
    # Mapping satellite names to their respective cloud properties
    cloud_properties = {
        "L5": "CLOUD_COVER",
        "L7": "CLOUD_COVER",
        "L8": "CLOUD_COVER",
        "L9": "CLOUD_COVER",
        "S2": "CLOUDY_PIXEL_PERCENTAGE",
    }
    cloud_property = cloud_properties.get(satellite)

    # Create a filter to select images with system:time_start month in the monthsToKeep list
    def filter_by_month(month):
        return ee.Filter.calendarRange(month, month, 'month') # type: ignore

    month_filters = [filter_by_month(month) for month in months_list]
    month_filter = ee.Filter.Or(month_filters) # type: ignore
    collection = (
        ee.ImageCollection(collection_name)
        .filterBounds(ee.Geometry.Polygon(polygon))
        .filterDate(ee.Date(start_date), ee.Date(end_date))
        .filterMetadata(cloud_property, "less_than", max_cloud_cover)
    )
    # apply the month filter to only include images from the months in the months_list
    collection = collection.filter(month_filter)
    
    return collection


def count_images_in_ee_collection(
    polygon: list[list[float]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    max_cloud_cover: float = 95,
    satellites: Collection[str] = ("L5", "L7", "L8", "L9", "S2"),
    tiers: list[int] = None,
    months_list:list[int] = None
) -> dict:
    """
    Count the number of images in specified satellite collections over a certain area and time period.

    Parameters:
    polygon (list[list[float]]): A list of lists representing the vertices of a polygon in lon/lat coordinates.
    start_date (str or datetime): The start date of the time period. If a string, it should be in 'YYYY-MM-DD' format.
    end_date (str or datetime): The end date of the time period. If a string, it should be in 'YYYY-MM-DD' format.
    max_cloud_cover (float, optional): The maximum cloud cover percentage. Images with a cloud cover percentage higher than this will be excluded. Defaults to 99.
    satellites (Collection[str], optional): A collection of satellite names. The function will return image counts for these satellites. Defaults to ("L5","L7","L8","L9","S2").
    tiers (list[int], optional): A list of tiers. The function will return image counts for these tiers. Defaults to [1,2]
    months_list (list[int], optional): A list of months to filter the images by. Defaults to None meaning all the months will be included.
    Returns:
    dict: A dictionary where the keys are the satellite names and the values are the image counts.

    Raises:
    ValueError: If start_date or end_date are not strings or datetime objects.

    Example:
    >>> polygon = [[[151.2957545, -33.7390216],
    ... [151.312234, -33.7390216],
    ... [151.312234, -33.7012561],
    ... [151.2957545, -33.7012561],
    ... [151.2957545, -33.7390216]]]
    >>> start_date = '2017-12-01'
    >>> end_date = '2018-01-01'
    >>> count_images(polygon, start_date, end_date)
    """
    if months_list is None:
        months_list = [1,2,3,4,5,6,7,8,9,10,11,12]
    # Check types of start_date and end_date
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    elif not isinstance(start_date, datetime):
        raise ValueError("start_date must be a string or datetime object")

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    elif not isinstance(end_date, datetime):
        raise ValueError("end_date must be a string or datetime object")
    
    # Check that end_date is after start_date
    if end_date <= start_date:
        raise ValueError(f"End date: {end_date.strftime('%Y-%m-%d')} must be after start date: {start_date.strftime('%Y-%m-%d')}")

    # Check if EE was initialized or not
    try:
        ee.ImageCollection("LANDSAT/LT05/C01/T1_TOA")
    except:
        ee.Initialize()

    if tiers is None:
        tiers = [1, 2]

    image_counts = {}
    images_in_tier_count = 0
    for satellite in satellites:
        images_in_tier_count = 0
        for tier in tiers:
            collection = get_collection_by_tier(
                polygon, start_date, end_date, satellite, tier, max_cloud_cover,months_list=months_list
            )
            if collection:
                images_in_tier_count += collection.size().getInfo()
        image_counts[satellite] = images_in_tier_count

    return image_counts


def download_url_dict(url_dict):
    for save_path, url in url_dict.items():
        # get a response from the url
        response = common.get_response(url, stream=True)
        with response:
            logger.info(f"response: {response}")
            logger.info(f"response.status_code: {response.status_code}")
            logger.info(f"response.headers: {response.headers}")
            if response.status_code == 404:
                logger.info(f"404 response for {url}")
                raise Exception(
                    f"404 response for {url}. Please raise an issue on GitHub."
                )

            # too many requests were made to the API
            if response.status_code == 429:
                content = response.text()
                print(
                    f"Response from API for status_code: {response.status_code}: {content}"
                )
                logger.info(
                    f"Response from API for status_code: {response.status_code}: {content}"
                )
                raise Exception(
                    f"Response from API for status_code: {response.status_code}: {content}"
                )

            # raise an exception if the response status_code is not 200
            if response.status_code != 200:
                print(f"response.status_code {response.status_code} for {url}")
                logger.info(f"response.status_code {response.status_code} for {url}")
                return False

            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                content_length = int(content_length)
                with open(save_path, "wb") as fd:
                    with tqdm.auto.tqdm(
                        total=content_length,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {os.path.basename(save_path)}",
                        initial=0,
                        ascii=False,
                        position=0,
                    ) as pbar:
                        for chunk in response.iter_content(1024):
                            if not chunk:
                                break
                            fd.write(chunk)
                            pbar.update(len(chunk))
            else:
                with open(save_path, "wb") as fd:
                    for chunk in response.iter_content(1024):
                        fd.write(chunk)


async def async_download_url_dict(url_dict: dict = {}):
    """
    Asynchronously downloads files from a given dictionary of URLs and save locations.

    Parameters
    ----------
    url_dict : dict, optional
        A dictionary where the keys represent local save paths and the values are the corresponding URLs of the files to be downloaded. Default is an empty dictionary.

    Usage
    -----
    url_dict = {
        "/path/to/save/file1.h5": "https://zenodo.org/record/7574784/file1.h5",
        "/path/to/save/file2.json": "https://zenodo.org/record/7574784/file2.json",
        "/path/to/save/file3.txt": "https://zenodo.org/record/7574784/file3.txt",
    }

    await async_download_url_dict(url_dict)
    """

    def session_creator():
        # Set the custom timeout value (in seconds)
        keepalive_timeout = 100
        # Configure the timeout
        connector = aiohttp.TCPConnector(keepalive_timeout=keepalive_timeout)
        # Create and return the session with the configured timeout
        return aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=600)
        )

    # allow 1 concurrent downloads
    semaphore = asyncio.Semaphore(1)
    tasks = []
    for save_path, url in url_dict.items():
        task = asyncio.create_task(
            download_zenodo_file(
                semaphore, session_creator, url, save_path, max_retries=0
            )
        )
        tasks.append(task)
    # start all the tasks at once
    await tqdm.asyncio.tqdm.gather(*tasks)


async def async_download_url_dict(url_dict: dict = {}):
    """
    Asynchronously downloads files from a given dictionary of URLs and save locations.

    Parameters
    ----------
    url_dict : dict, optional
        A dictionary where the keys represent local save paths and the values are the corresponding URLs of the files to be downloaded. Default is an empty dictionary.

    Usage
    -----
    url_dict = {
        "/path/to/save/file1.h5": "https://zenodo.org/record/7574784/file1.h5",
        "/path/to/save/file2.json": "https://zenodo.org/record/7574784/file2.json",
        "/path/to/save/file3.txt": "https://zenodo.org/record/7574784/file3.txt",
    }

    await async_download_url_dict(url_dict)
    """

    def session_creator():
        # Set the custom timeout value (in seconds)
        keepalive_timeout = 100
        # Configure the timeout
        connector = aiohttp.TCPConnector(keepalive_timeout=keepalive_timeout)
        # Create and return the session with the configured timeout
        return aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=600)
        )

    # allow 1 concurrent downloads
    semaphore = asyncio.Semaphore(1)
    tasks = []
    for save_path, url in url_dict.items():
        task = asyncio.create_task(
            download_zenodo_file(
                semaphore, session_creator, url, save_path, max_retries=0
            )
        )
        tasks.append(task)
    # start all the tasks at once
    await tqdm.asyncio.tqdm.gather(*tasks)


async def download_zenodo_file(
    semaphore: asyncio.Semaphore,
    session_creator: callable,
    url: str,
    save_location: str,
    max_retries: int = 1,
):
    """
    Asynchronously downloads a file from Zenodo, given a URL and save location, with a specified maximum number of retries.

    Parameters
    ----------
    semaphore : asyncio.Semaphore
        An asyncio Semaphore object used to limit the number of concurrent downloads.
    session_creator : callable
        A function that creates and returns an aiohttp.ClientSession object.
    url : str
        The URL of the file to be downloaded.
    save_location : str
        The local path where the downloaded file should be saved.
    max_retries : int, optional
        The maximum number of times the download should be retried if it fails, default is 1.

    Returns
    -------
    bool
        True if the download is successful, otherwise raises an Exception.

    Raises
    ------
    Exception
        If the download fails after the specified number of retries.

    Usage
    -----
    semaphore = asyncio.Semaphore(10)
    session_creator = ...
    url = "https://zenodo.org/record/12345/files/myfile.zip"
    save_location = "/path/to/save/myfile.zip"
    max_retries = 3

    await download_zenodo_file(semaphore, session_creator, url, save_location, max_retries)
    """
    async with session_creator() as session:
        is_download_success, retry_after, status_code = await download_with_retry(
            semaphore, session, url, save_location
        )
        logger.info(
            f"is_download_success, retry_after,status_code {is_download_success, retry_after,status_code}"
        )
        retry_count = 0
        while not is_download_success and retry_count < max_retries:
            # If the session was closed, create a new one
            if session.closed:
                async with session_creator() as session:
                    logger.warning(
                        f"Download failed. Retrying ({retry_count + 1}/{max_retries})..."
                    )
                    if retry_after is not None:
                        await asyncio.sleep(retry_after)
                    (
                        is_download_success,
                        retry_after,
                        status_code,
                    ) = await download_with_retry(
                        semaphore, session, url, save_location
                    )
                    logger.info(
                        f"is_download_success, retry_after,status_code {is_download_success, retry_after,status_code}"
                    )
                    retry_count += 1
            else:
                logger.warning(
                    f"Download failed. Retrying ({retry_count + 1}/{max_retries})..."
                )
                if retry_after is not None:
                    await asyncio.sleep(retry_after)
                (
                    is_download_success,
                    retry_after,
                    status_code,
                ) = await download_with_retry(semaphore, session, url, save_location)
                logger.info(
                    f"is_download_success, retry_after,status_code {is_download_success, retry_after,status_code}"
                )
                retry_count += 1

        if not is_download_success:
            raise Exception(
                f"Download failed for {save_location} after {max_retries} retries for status code {status_code}. Please try again later."
            )

        return is_download_success


async def download_with_retry(
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    save_location: str,
    rate_limit_remaining=None,
) -> Tuple[bool, Optional[int]]:
    """
    Asynchronously downloads a file given a URL and save location, with rate limiting and error handling.

    Parameters
    ----------
    semaphore : asyncio.Semaphore
        An asyncio Semaphore object used to limit the number of concurrent downloads.
    session : aiohttp.ClientSession
        An aiohttp ClientSession object used for making HTTP requests.
    url : str
        The URL of the file to be downloaded.
    save_location : str
        The local path where the downloaded file should be saved.
    rate_limit_remaining : Optional[int], optional
        The remaining rate limit, if any; default is None.
        This is used to determine whether to wait before retrying a download.

    Returns
    -------
    Tuple[bool, Optional[int]]
        A tuple containing:
        - A boolean indicating whether the download was successful.
        - An optional integer specifying the number of seconds to wait before retrying in case of a rate limit or a failed download.
        - An integer representing the status code of the HTTP response.

    Usage
    -----
    semaphore = asyncio.Semaphore(10)
    session = aiohttp.ClientSession()
    url = "https://example.com/file.zip"
    save_location = "/path/to/save/file.zip"

    is_download_success, retry_after, status_code = await download_with_retry(semaphore, session, url, save_location)
    """
    logger.info(f"rate_limit_remaining: {rate_limit_remaining}")
    logger.info(f"session: {session} is session closed {session.closed}")
    async with semaphore:
        logger.info(f"semaphore was accessed {semaphore}")
        async with session.get(url) as response:
            logger.info(f"response: {response}")
            if rate_limit_remaining is None:
                rate_limit_remaining = int(
                    response.headers.get("X-RateLimit-Remaining", -1)
                )
            logger.info(f"rate_limit_remaining: {rate_limit_remaining}")
            logger.info(f"response.status: {response.status}")
            logger.info(f"response.headers: {response.headers}")

            if response.status == 404:
                logger.info(f"404 response for {url}")
                raise Exception(
                    f"404 response for {url}. Please raise an issue on GitHub."
                )

            # this means X-RateLimit-Remaining was close to 0 and we need to wait
            if rate_limit_remaining == 1 or response.status == 429:
                content = await response.text()
                logger.info(
                    f"Response from API for status: {response.status}: {content}"
                )
                # by default, wait for 60 seconds or the number of seconds specified in the Retry-After header
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.info(f"retry_after: {retry_after}")
                return False, retry_after, response.status

            # raise an exception if the response status is not 200
            if response.status != 200:
                logger.info(f"response.status {response.status} for {url}")
                return False, 1, response.status
            # response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                content_length = int(content_length)
                with open(save_location, "wb") as fd:
                    with tqdm.auto.tqdm(
                        total=content_length,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {os.path.basename(save_location)}",
                        initial=0,
                        ascii=False,
                        position=0,
                    ) as pbar:
                        async for chunk in response.content.iter_chunked(1024):
                            if not chunk:
                                break
                            fd.write(chunk)
                            pbar.update(len(chunk))
                        return True, None, response.status
            else:
                with open(save_location, "wb") as fd:
                    async for chunk in response.content.iter_chunked(1024):
                        fd.write(chunk)
                    return True, None, response.status


async def async_download_url(session, url: str, save_path: str):
    model_name = url.split("/")[-1]
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


async def download_file(session, url, save_location):
    retries = 2  # number of times to retry download
    for i in range(retries):
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"An error occurred while downloading.{response}")
                    logger.error(f"An error occurred while downloading.{response}")
                    print(response.status)
                    return
                with open(save_location, "wb") as f:
                    async for chunk in response.content.iter_chunked(1024):
                        if not chunk:
                            break
                        f.write(chunk)
                break  # break out of retry loop if download is successful
        except asyncio.exceptions.TimeoutError as e:
            logger.error(e)
            logger.error(f"An error occurred while downloading {save_location}.{e}")
            print(
                f"Timeout error occurred for {url}. Retrying with new session in 1 second... ({i + 1}/{retries})"
            )
            logger.warning(
                f"Timeout error occurred for {url}. Retrying with new session in 1 second... ({i + 1}/{retries})"
            )
            await asyncio.sleep(1)
            async with aiohttp.ClientSession() as new_session:
                return await download_file(new_session, url, save_location)
        except Exception as e:
            logger.error(e)
            logger.error(
                f"Download failed for {save_location} {url}. Retrying in 1 second... ({i + 1}/{retries})"
            )
            logger.warning(
                f"Timeout error occurred for {url}. Retrying with new session in 1 second... ({i + 1}/{retries})"
            )
            print(
                f"Download failed for {url}. Retrying in 1 second... ({i + 1}/{retries})"
            )
            await asyncio.sleep(1)
    else:
        logger.error(f"Download failed for {save_location} {url}.")
        print(f"Download failed for {url}.")
        return


async def download_group(session, group, semaphore):
    coroutines = []
    logger.info(f"group: {group}")
    for tile_number, tile in enumerate(group):
        polygon = tile["polygon"]
        filepath = os.path.abspath(tile["filepath"])
        filenames = {
            "multiband": "multiband" + str(tile_number),
        }
        for tile_id in tile["ids"]:
            logger.info(f"tile_id: {tile_id}")
            file_id = tile_id.replace("/", "_")
            filename = filenames["multiband"] + "_" + file_id
            save_location = os.path.join(filepath, filename.replace("/", "_") + ".zip")
            logger.info(f"save_location: {save_location}")
            coroutines.append(
                async_download_tile(
                    session,
                    polygon,
                    tile_id,
                    save_location,
                    filename,
                    filePerBand=False,
                    semaphore=semaphore,
                )
            )

    year_name = os.path.basename(group[0]["filepath"])
    await tqdm.asyncio.tqdm.gather(
        *coroutines, leave=False, desc=f"Downloading {year_name}"
    )
    # await asyncio.gather(*coroutines)

    logger.info(f"Files downloaded to {group[0]['filepath']}")
    common.unzip_dir(group[0]["filepath"])
    common.delete_empty_dirs(group[0]["filepath"])
    # delete duplicate tifs. keep tif with most non-black pixels
    # common.delete_tifs_at_same_location(group[0]["filepath"])


# Download the information for each year
async def download_groups(
    groups,
    semaphore: asyncio.Semaphore,
    group_id: str = "",
    show_progress_bar: bool = False,
):
    coroutines = []
    async with aiohttp.ClientSession() as session:
        logger.info(f"group: {groups}")
        for key, group in groups.items():
            logger.info(f"key: {key} group: {group}")
            if len(group) > 0:
                coroutines.append(download_group(session, group, semaphore))
            else:
                print(f"No tiles available to download for year: {key}")
                logger.warning(f"No tiles available to download for year: {key}")

        if show_progress_bar == False:
            await asyncio.gather(*coroutines)
        elif show_progress_bar == True:
            await tqdm.asyncio.tqdm.gather(
                *coroutines,
                position=1,
                leave=False,
                desc=f"Downloading years for ROI {group_id}",
            )


async def download_ROIs(ROI_tiles: dict = {}):
    tasks = []
    semaphore = asyncio.Semaphore(15)
    show_progress_bar = True
    for ROI_id, ROI_info in ROI_tiles.items():
        tasks.append(
            download_groups(
                ROI_info,
                semaphore,
                group_id=ROI_id,
                show_progress_bar=show_progress_bar,
            )
        )
    await tqdm.asyncio.tqdm.gather(*tasks, position=0, desc=f"Downloading ROIs")


async def async_download_tile(
    session: aiohttp.ClientSession,
    polygon: List[set],
    tile_id: str,
    filepath: str,
    filename: str,
    filePerBand: bool,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Download a single tile of an Earth Engine image and save it to a zip directory.

    This function uses the Earth Engine API to crop the image to a specified polygon and download it to a zip directory with the specified filename. The number of concurrent downloads is limited to 10.

    Parameters:

    session (aiohttp.ClientSession): An instance of aiohttp session to make the download request.
    polygon (List[set]): A list of latitude and longitude coordinates that define the region to crop the image to.
    tile_id (str): The ID of the Earth Engine image to download.
    filepath (str): The path of the directory to save the downloaded zip file to.
    filename (str): The name of the zip file to be saved.
    filePerBand (bool): Whether to save each band of the image in a separate file or as a single file.
    semaphore:asyncio.Semaphore : Limits number of concurrent requests
    Returns:
    None
    """
    # Semaphore limits number of concurrent requests
    async with semaphore:
        OUT_RES_M = 0.5  # output raster spatial footprint in metres
        image_ee = ee.Image(tile_id)
        # crop and download
        download_id = ee.data.getDownloadId(
            {
                "image": image_ee,
                "region": polygon,
                "scale": OUT_RES_M,
                "crs": "EPSG:4326",
                "filePerBand": filePerBand,
                "name": filename,
            }
        )
        try:
            # create download url using id
            url = ee.data.makeDownloadUrl(download_id)
            await download_file(session, url, filepath)
        except Exception as e:
            logger.error(e)
            raise e


def run_async_function(async_callback, **kwargs) -> None:
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # apply a nested loop to jupyter's event loop for async downloading
    nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_callback(**kwargs))
    logger.info(f"result: {result}")
    return result


def get_num_splitters(gdf: gpd.GeoDataFrame) -> int:
    """
    Calculates the minimum number of splitters required to divide a geographic region represented by a GeoDataFrame into smaller, equal-sized tiles whose
    area <= 1 km^2.

    max area per tile is 1 km^2

    Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame representing the geographic region to be split. Must contain only a single entry.

    Returns:
        int: An integer representing the minimum number of splitters required to divide the region represented by `gdf` into smaller, equal-sized tiles whose
        area <= 1 km^2.

    """
    # convert to geojson dictionary
    logger.info(f"gdf: {gdf}")
    roi_json = gdf.to_json()
    logger.info(f"roi_json: {roi_json}")
    roi_json = json.loads(roi_json)
    # only one feature is present select 1st feature's geometry
    roi_geometry = roi_json["features"][0]["geometry"]
    # get area of entire shape as squared kilometers
    area_km2 = area(roi_geometry) / 1e6
    logger.info(f"Area: {area_km2}")
    if area_km2 <= 1:
        return 0
    # get minimum number of horizontal and vertical splitters to split area equally
    # max area per tile is 1 km^2
    num_splitters = math.ceil(math.sqrt(area_km2))
    return num_splitters


def splitPolygon(polygon: gpd.GeoDataFrame, num_splitters: int) -> MultiPolygon:
    """
    Split a polygon into a given number of smaller polygons by adding horizontal and vertical lines.

    Parameters:
    polygon (gpd.GeoDataFrame): A GeoDataFrame object containing a single polygon.
    num_splitters (int): The number of horizontal and vertical lines to add.

    Returns:
    MultiPolygon: A MultiPolygon object containing the smaller polygons.

    Example:
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon, MultiPolygon
    >>> poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    >>> df = gpd.GeoDataFrame(geometry=[poly])
    >>> result = splitPolygon(df, 2)
    >>> result # polygon split into 4 equally sized tiles
    """
    minx, miny, maxx, maxy = polygon.bounds.iloc[0]
    dx = (maxx - minx) / num_splitters  # width of a small part
    dy = (maxy - miny) / num_splitters  # height of a small part
    horizontal_splitters = [
        LineString([(minx, miny + i * dy), (maxx, miny + i * dy)])
        for i in range(num_splitters)
    ]
    vertical_splitters = [
        LineString([(minx + i * dx, miny), (minx + i * dx, maxy)])
        for i in range(num_splitters)
    ]
    splitters = horizontal_splitters + vertical_splitters
    result = polygon["geometry"].iloc[0]
    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))
    return result


def remove_zip(path) -> None:
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
    # Remove each zip file
    for zipped_file in zipped_files:
        os.remove(zipped_file)


def unzip(path) -> None:
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
    # Unzip each file
    for zipped_file in zipped_files:
        with zipfile.ZipFile(zipped_file, "r") as zip_ref:
            zip_ref.extractall(path)


def unzip_files(paths):
    # Create a thread pool with a fixed number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a unzip task for each directory
        futures = [executor.submit(unzip, path) for path in paths]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


def remove_zip_files(paths):
    # Create a thread pool with a fixed number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a remove_zip task for each directory
        futures = [executor.submit(remove_zip, path) for path in paths]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


def get_subdirs(parent_dir: str):
    # Get a list of all the subdirectories in the parent directory
    subdirectories = []
    for root, dirs, files in os.walk(parent_dir):
        for d in dirs:
            subdirectories.append(os.path.join(root, d))
    return subdirectories


def create_dir(dir_path: str, raise_error=True) -> str:
    dir_path = os.path.abspath(dir_path)
    if os.path.exists(dir_path):
        if raise_error:
            raise FileExistsError(dir_path)
    else:
        os.makedirs(dir_path)
    return dir_path


def copy_multiband_tifs(roi_path: str, multiband_path: str):
    for folder in glob(
        roi_path + os.sep + "tile*",
        recursive=True,
    ):
        files = glob(folder + os.sep + "*multiband.tif")
        [
            shutil.copyfile(file, multiband_path + os.sep + file.split(os.sep)[-1])
            for file in files
        ]


def mk_filepaths(tiles_info: List[dict]):
    """
    Copy multiband TIF files from a source folder to a destination folder.

    This function uses the glob module to search for multiband TIF files in the subfolders of a source folder, and then uses the shutil module to copy each file to a destination folder. The name of the file in the destination folder is set to the name of the file in the source folder.

    Parameters:

    roi_path (str): The path of the source folder to search for multiband TIF files.
    multiband_path (str): The path of the destination folder to copy the multiband TIF files to.
    Returns:
    None"""
    filepaths = [tile_info["filepath"] for tile_info in tiles_info]
    for filepath in filepaths:
        create_dir(filepath, raise_error=False)


def create_tasks(
    session: aiohttp.ClientSession,
    polygon: List[tuple],
    tile_id: str,
    filepath: str,
    filename: str,
) -> list:
    """

    creates a list of tasks that are used to download the data.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The aiohttp.ClientSession object that handles the connection with the
        api.
    polygon : List[tuple] coordinates of the polygon in lat/lon
    ex: [(1,2),(3,4)(4,3),(3,3),(1,2)]
    tile_id : str
        GEE id of the tile
        ex: 'USDA/NAIP/DOQQ/m_4012407_se_10_1_20100612'
    filepath : str
        full path to tile directory that data will be saved to
    filename : str
       name of file that data will be downloaded to
    file_id : str
        name that file will be saved as based on tile_id

    Returns
    -------
    tasks : list
        A list of tasks that are used to download the data.

    """
    tasks = []
    task = asyncio.create_task(
        async_download_tile(
            session,
            polygon,
            tile_id,
            filepath,
            filename,
            filePerBand=False,
        )
    )
    tasks.append(task)
    return tasks


async def async_download_all_tiles(tiles_info: List[dict]) -> None:
    # creates task for each tile to be downloaded and waits for tasks to complete
    tasks = []
    for counter, tile_dict in enumerate(tiles_info):
        polygon = tile_dict["polygon"]
        filepath = os.path.abspath(tile_dict["filepath"])
        parent_dir = os.path.dirname(filepath)
        multiband_filepath = os.path.join(parent_dir, "multiband")
        async with aiohttp.ClientSession(timeout=3000) as session:
            for tile_id in tile_dict["ids"]:
                logger.info(f"tile_id: {tile_id}")
                file_id = tile_id.replace("/", "_")
                # handles edge case where tile has 2 years in the tile ID by extracting the ealier year
                year_str = file_id.split("_")[-1][:4]
                if len(file_id.split("_")[-2]) == 8:
                    year_str = file_id.split("_")[-2][:4]

                filename = "multiband" + str(counter) + "_" + file_id
                # full path to year directory within multiband dir eg. ./multiband/2012
                year_filepath = os.path.join(multiband_filepath, year_str)
                logger.info(f"year_filepath: {year_filepath}")
                tasks.extend(
                    create_tasks(
                        session,
                        polygon,
                        tile_id,
                        filepath,
                        year_filepath,
                        filename,
                        file_id,
                    )
                )
    # show a progress bar of all the requests in progress
    await tqdm.asyncio.tqdm.gather(*tasks, position=0, desc=f"All Downloads")


async def get_ids_for_tile(
    year_path, gee_collection, tile, dates, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        collection = ee.ImageCollection(gee_collection)
        polygon = ee.Geometry.Polygon(tile)
        # Filter the collection to get only the images within the tile and date range
        filtered_collection = (
            collection.filterBounds(polygon)
            .filterDate(*dates)
            .sort("system:time_start", True)
        )
        # Get a list of all the image names in the filtered collection
        image_list = filtered_collection.getInfo().get("features")
        ids = [obj["id"] for obj in image_list]
        # Create a dictionary for each tile with the information about the images to be downloaded
        image_dict = {
            "polygon": tile,
            "ids": ids,
            "filepath": year_path,
        }
        return image_dict


async def get_tiles_info(
    tile_coords: list,
    dates: Tuple[str],
    roi_path: str,
    gee_collection: str,
    semaphore: asyncio.Semaphore,
) -> List[dict]:
    """
    Get information about images within the specified tile coordinates, date range, and image collection.
    The information includes the image IDs, file path, and polygon geometry of each tile.

    Parameters:
    tile_coords (List[List[Tuple[float]]]): A list of tile coordinates, where each tile coordinate is a list of
                                            (latitude, longitude) tuples that define the polygon of the tile.
    dates (Tuple[str]): A tuple of two strings representing the start and end dates for the image collection.
    roi_path (str): The path to the directory where the images will be saved.
    gee_collection (str): The name of the image collection on Google Earth Engine.

    Returns:
    List[dict]: A list of dictionaries, where each dictionary contains information about a single tile, including the
                polygon geometry, the IDs of the images within the tile, and the file path to the directory where the
                images will be saved.
    """
    logger.info(f"dates: {dates}")
    logger.info(f"roi_path: {roi_path}")
    logger.info(f"tile_coords: {tile_coords}")
    start_year = dates[0].strftime("%Y")
    year_path = os.path.join(roi_path, "multiband", start_year)
    tasks = []
    for tile in tile_coords:
        tasks.append(
            get_ids_for_tile(year_path, gee_collection, tile, dates, semaphore)
        )

    # create a progress bar for the number of tasks
    pbar = tqdm.asyncio.tqdm(
        total=len(tasks), position=0, desc=f"Getting tiles for year {start_year}"
    )

    # iterate over completed tasks using asyncio.as_completed()
    results = []
    for coro in asyncio.as_completed(tasks):
        # update progress bar
        pbar.update(1)
        result = await coro
        logger.info(f"coro result {result}")
        results.append(result)

    # close progress bar and return results
    pbar.close()
    return {start_year: results}

    # # results = await asyncio.gather(*tasks)
    # start_year = dates[0].strftime("%Y")
    # results = await tqdm.asyncio.tqdm.gather(
    #     *tasks, position=0,desc=f"Getting tiles for year {start_year}"
    # )
    # return {start_year: results}


def get_tile_coords(num_splitters: int, roi_gdf: gpd.GeoDataFrame) -> list[list[list]]:
    """
    Given the number of splitters and a GeoDataFrame,Splits an ROI geodataframe into tiles of 1km^2 area (or less),
    and returns a list of lists of tile coordinates.

    Args:
        num_splitters (int): The number of splitters to divide the ROI into.
        gpd_data (gpd.GeoDataFrame): The GeoDataFrame containing the ROI geometry.

    Returns:
        list[list[list[float]]]: A list of lists of tile coordinates, where each inner list represents the coordinates of one tile.
        The tile coordinates are in [lat,lon] format.
    """
    if num_splitters == 0:
        split_polygon = Polygon(roi_gdf["geometry"].iloc[0])
        tile_coords = [list(split_polygon.exterior.coords)]
    elif num_splitters > 0:
        # split ROI into rectangles of 1km^2 area (or less)
        split_polygon = splitPolygon(roi_gdf, num_splitters)
        tile_coords = [list(part.exterior.coords) for part in split_polygon.geoms]
    return tile_coords


def create_ROI_directories(download_path: str, roi_id: str, dates):
    """
    Creates directories to store downloaded data for a single region of interest (ROI).

    The function creates a main directory for the ROI, a subdirectory for multiband files, and subdirectories for each year in the date range.

    Parameters:
    - download_path (str): The path to the directory where downloaded data should be stored.
    - roi_id (str): The ID of the ROI.
    - dates (List[str]): A list containing the start and end dates of the date range in the format ['YYYY-MM-DD', 'YYYY-MM-DD'].

    Returns:
    - None
    """
    # name of ROI folder to contain all downloaded data
    roi_name = f"ID_{roi_id}_dates_{dates[0]}_to_{dates[1]}"
    roi_path = os.path.join(download_path, roi_name)
    # create directory to hold all multiband files
    multiband_path = file_utilities.create_directory(roi_path, "multiband")
    # create subdirectories for each year
    start_date = dates[0].split("-")[0]
    end_date = dates[1].split("-")[0]
    logger.info(f"start_date : {start_date } end_date : {end_date }")
    common.create_year_directories(int(start_date), int(end_date), multiband_path)
    return roi_path


def prepare_ROI_for_download(
    download_path: str,
    roi_gdf: gpd.GeoDataFrame,
    ids: List[str],
    dates: Tuple[str],
) -> None:
    for roi_id in ids:
        create_ROI_directories(download_path, roi_id, dates)
        roi_gdf = roi_gdf.loc[id]


async def get_tiles_info_per_year(
    roi_path: str,
    rois_gdf: gpd.GeoDataFrame,
    roi_id: str,
    dates: Tuple[str],
    semaphore: asyncio.Semaphore,
):
    gee_collection = "USDA/NAIP/DOQQ"
    logger.info(f"rois_gdf : {rois_gdf }")
    gdf = rois_gdf.loc[[roi_id]]
    logger.info(f"gdf : {gdf }")
    roi_gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry.name)
    logger.info(f"roi_gdf : {roi_gdf }")
    # get number of splitters need to split ROI into rectangles of 1km^2 area (or less)
    num_splitters = get_num_splitters(roi_gdf)
    logger.info(f"Splitting ROI into {num_splitters}x{num_splitters} tiles")

    # split ROI into rectangles of 1km^2 area (or less)
    tile_coords = get_tile_coords(num_splitters, roi_gdf)
    logger.info(f"tile_coords: {tile_coords}")
    yearly_ranges = common.get_yearly_ranges(dates)

    tiles_per_year = {}
    tasks = []
    # for each year get the tiles available to download
    for year_date in yearly_ranges:
        tasks.append(
            get_tiles_info(tile_coords, year_date, roi_path, gee_collection, semaphore)
        )
    # list_of_tiles = await asyncio.gather(*tasks)
    # create a progress bar for the number of tasks
    # pbar = tqdm.asyncio.tqdm(total=len(tasks), desc=f"Downloading tiles for ROI: {roi_id}")

    # iterate over completed tasks using asyncio.as_completed()
    list_of_tiles = []
    for task in asyncio.as_completed(tasks):
        result = await task
        tile_key = list(result.keys())[0]
        tiles_per_year[tile_key] = result[tile_key]
        list_of_tiles.append(result)

    logger.info(f"tiles_per_year: {tiles_per_year}")
    return {roi_id: tiles_per_year}


# call asyncio to run download_ROIs
async def get_tiles_for_ids(
    roi_paths: str,
    rois_gdf: gpd.GeoDataFrame,
    selected_ids: List[str],
    dates: Tuple[str],
) -> None:
    """creates a nested loop that's used to asynchronously download imagery and waits for all the imagery to download
    Args:
        download_path (str): full path to directory to download imagery to
        roi_gdf (gpd.GeoDataFrame): geodataframe of ROIs on the map
        ids (List[str]): ids of ROIs to download imagery for
        dates (Tuple[str]): start and end dates
    """
    ROI_tiles = {}
    tasks = []
    semaphore = asyncio.Semaphore(50)

    for roi_path, roi_id in zip(roi_paths, selected_ids):
        tasks.append(
            get_tiles_info_per_year(roi_path, rois_gdf, roi_id, dates, semaphore)
        )

    list_of_ROIs = await tqdm.asyncio.tqdm.gather(
        *tasks, position=0, desc=f"Getting tiles for each ROI"
    )

    for roi in list_of_ROIs:
        roi_id = list(roi.keys())[0]
        ROI_tiles[roi_id] = roi[roi_id]

    logger.info(f"ROI_tiles: {ROI_tiles}")
    for roi_id in ROI_tiles.keys():
        for year in ROI_tiles[roi_id].keys():
            mk_filepaths(ROI_tiles[roi_id][year])

    return ROI_tiles


async def async_download_ROIs(ROI_tiles: List[dict]) -> None:
    """
    Downloads the specified bands for each ROI tile asynchronously using aiohttp and asyncio.

    Parameters:
    ROI_tiles (List[dict]): A list of dictionaries representing the ROI tiles to download.
    Returns:
    None: This function does not return anything.
    """
    async with aiohttp.ClientSession() as session:
        # creates task for each tile to be downloaded and waits for tasks to complete
        tasks = []
        for ROI_tile in ROI_tiles:
            for year in ROI_tile.keys():
                print(f"YEAR for ROI tile: {year}")
                task = asyncio.create_task(async_download_year(ROI_tile[year], session))
                tasks.append(task)
        # show a progress bar of all the requests in progress
        await tqdm.asyncio.tqdm.gather(*tasks, position=0, desc=f"All Downloads")


async def async_download_year(tiles_info: List[dict], session) -> None:
    """
    Downloads the specified bands for each tile in the specified year asynchronously using aiohttp and asyncio.

    Parameters:
    tiles_info (List[dict]): A list of dictionaries representing the tiles to download.
    session: The aiohttp session to use for downloading.

    Returns:
    None: This function does not return anything.
    """
    # creates task for each tile to be downloaded and waits for tasks to complete
    tasks = []
    for counter, tile_dict in enumerate(tiles_info):
        polygon = tile_dict["polygon"]
        filepath = os.path.abspath(tile_dict["filepath"])
        filenames = {
            "multiband": "multiband" + str(counter),
        }
        for tile_id in tile_dict["ids"]:
            logger.info(f"tile_id: {tile_id}")
            file_id = tile_id.replace("/", "_")
            logger.info(f"year_filepath: {filepath}")
            tasks.extend(
                create_tasks(
                    session,
                    polygon,
                    tile_id,
                    filepath,
                    filepath,
                    filenames,
                    file_id,
                )
            )
    # show a progress bar of all the requests in progress
    await tqdm.asyncio.tqdm.gather(*tasks, position=0, desc=f"All Downloads")
    common.unzip_data(os.path.dirname(filepath))
    # delete any directories that were empty
    common.delete_empty_dirs(os.path.dirname(filepath))
