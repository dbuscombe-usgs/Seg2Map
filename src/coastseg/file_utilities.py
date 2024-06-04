import os
import re
import glob
import shutil
import json
import logging
import datetime
from typing import Union, Collection

# Specific classes/functions from modules
from typing import List, Union
from json import JSONEncoder
from contextlib import contextmanager
from tqdm.auto import tqdm

# Third-party imports
import geopandas as gpd
import geojson
import numpy as np
from coastseg import core_utilities

# Logger setup
logger = logging.getLogger(__name__)


@contextmanager
def progress_bar_context(
    use_progress_bar: bool, total: int = 6, description: str = "", **kwargs
):
    """
    Context manager for handling progress bar creation and updates.

    Parameters:
    -----------
    use_progress_bar : bool
        If True, a tqdm progress bar will be displayed. Otherwise, it's a no-op.

    Yields:
    -------
    function
        A function to call for updating the progress bar. If use_progress_bar is False,
        it's a no-op.
    """
    if use_progress_bar:
        progress_bar = tqdm(total=total, dynamic_ncols=True, desc=description, **kwargs)

        def update(message: str, update_value: float = 1):
            progress_bar.set_description(message)
            progress_bar.update(update_value)

        yield update
        progress_bar.close()
    else:
        # If not using a progress bar, just yield a no-op function
        yield lambda message: None

def get_ROI_ID_from_session(session_name: str) -> str:
    """
    Retrieves the ROI ID from the config.json file in the extracted shoreline session directory.

    Args:
        session_name (str): The name of the session.

    Returns:
        str: The ROI ID.

    Raises:
        Exception: If the session directory does not exist.
    """
    # need to read the ROI ID from the config.json file found in the extracted shoreline session directory
    session_directory = os.path.join(core_utilities.get_base_dir(), "sessions", session_name)
    if not os.path.exists(session_directory):
        raise Exception(f"The session directory {session_directory} does not exist")
    config_json_location = find_file_recursively(
        session_directory, "config.json"
    )
    config = load_data_from_json(config_json_location)
    roi_id = config.get("roi_id", "")
    return roi_id

def load_package_resource(
    resource_name: str,
    file_name: str = "",
    pkg_name: str = "coastseg",
) -> str:
    """
    Loads a resource from a package.

    Args:
        resource_name (str): Name of the resource to load.
        file_name (str): Specific file name within the resource, if applicable.
        pkg_name (str): Name of the package where the resource is located.

    Returns:
        str: Absolute path to the resource or file.

    Raises:
        ImportError: If necessary importlib resources are not available.
        FileNotFoundError: If the resource or file is not found.
    """
    try:
        from importlib import resources
    except (ImportError, AttributeError):
        try:
            # Use importlib backport for Python older than 3.9
            import importlib_resources as resources
        except ImportError:
            raise ImportError(
                "Both importlib.resources and importlib_resources are unavailable. Ensure you have the necessary packages installed."
            )
    # Get the resource location
    resource_location = resources.files(pkg_name).joinpath(resource_name)
    # Check if the resource exists and is a directory
    if not (resource_location.exists() and resource_location.is_dir()):
        raise FileNotFoundError(resource_location)
    # If a file name is provided, update the resource location and check again
    if file_name:
        resource_location = resource_location.joinpath(file_name)
        if not resource_location.exists():
            raise FileNotFoundError(resource_location)

    return os.path.abspath(resource_location)


def directory_exists(directory_name):
    return os.path.isdir(directory_name)


def generate_datestring() -> str:
    """Returns a datetime string in the following format %m-%d-%y__%I_%M_%S
    EX: "ID_0__01-31-22_12_19_45"""
    date = datetime.datetime.now()
    return date.strftime("%m-%d-%y__%I_%M_%S")


def read_json_file(json_file_path: str, raise_error=False, encoding="utf-8") -> dict:
    """
    Reads a JSON file and returns the parsed data as a dictionary.

    Args:
        json_file_path (str): The path to the JSON file.
        encoding (str, optional): The encoding of the file. Defaults to "utf-8".
        raise_error (bool, optional): Set to True if an error should be raised if the file doesn't exist.

    Returns:
        dict: The parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist and `raise_error` is True.

    """
    if not os.path.exists(json_file_path):
        if raise_error:
            raise FileNotFoundError(
                f"Model settings file does not exist at {json_file_path}"
            )
        else:
            return {}
    with open(json_file_path, "r", encoding=encoding) as f:
        data = json.load(f)
    return data


def get_session_contents_location(session_name: str, roi_id: str = ""):
    session_path = get_session_location(session_name)
    if roi_id:
        roi_location = find_matching_directory_by_id(session_path, roi_id)
        if roi_location is not None:
            session_path = roi_location
        # check if a config file exists in the session if it doesn't then this isn't correct
        try:
            find_file_by_regex(session_path, r"^config\.json$")
        except FileNotFoundError:
            raise Exception(
                f"Session Directory didn't contains config json file {session_path}"
            )
    if not os.path.isdir(session_path):
        raise Exception(f"Session Directory didn't exist {session_path}")
    return session_path


def find_file_by_regex(
    search_path: str, search_pattern: str = r"^config\.json$"
) -> str:
    """Searches for a file with matching regex in the specified directory

    Args:
        search_path (str): the directory path to search for the  file matching the search pattern
        search_pattern (str): the regular expression pattern to search for the config file

    Returns:
        str: the file path that matched the search_pattern

    Raises:
        FileNotFoundError: if a file is not found in the specified directory
    """
    logger.info(f"searching directory for config : {search_path}")
    config_regex = re.compile(search_pattern, re.IGNORECASE)
    for file in os.listdir(search_path):
        if config_regex.match(file):
            logger.info(f"{file} matched regex")
            file_path = os.path.join(search_path, file)
            return file_path

    raise FileNotFoundError(
        f"file matching pattern {search_pattern} was not found at {search_path}"
    )


def validate_config_files_exist(src: str) -> bool:
    """Check if config files exist in the source directory.
    Looks for files with names starting with "config_gdf" and ending with ".geojson"
    and a file named "config.json" in the source directory.
    Args:
        src (str): the source directory
    Returns:
        bool: True if both files exist, False otherwise
    """
    files = os.listdir(src)
    config_gdf_exists = False
    config_json_exists = False
    for file in files:
        if file.startswith("config_gdf") and file.endswith(".geojson"):
            config_gdf_exists = True
        elif file == "config.json":
            config_json_exists = True
        if config_gdf_exists and config_json_exists:
            return True
    return False


def move_files(
    src: Union[str, Collection[str]], dst_dir: str, delete_src: bool = False
) -> None:
    """
    Moves every file from either a source directory or a list of source files
    to a destination directory and optionally deletes the source when finished.

    Args:
    - src (str or list): The source directory path or a list of source file paths.
    - dst_dir (str): The destination directory path.
    - delete_src (bool): Whether to delete the source directory or files after moving. Defaults to False.

    Returns:
    - None
    """

    # Validate src: it must be a directory path or a list of file paths.
    if isinstance(src, str):
        if os.path.isdir(src):
            src_files = [os.path.join(src, filename) for filename in os.listdir(src)]
            logger.info(
                f"Moving all files from directory {src} to {dst_dir}. Delete Source: {delete_src}"
            )
        elif os.path.isfile(src):
            src_files = [src]
            logger.info(f"Moving file {src} to {dst_dir}. Delete Source: {delete_src}")
        else:
            logger.error(
                f"Provided src is a string but not a valid directory path: {src}"
            )
            return
    elif isinstance(src, list):
        src_files = src
        logger.info(f"Moving listed files to {dst_dir}. Delete Source: {delete_src}")
    else:
        logger.error(
            "The src parameter must be a directory path (str) or a list of file paths (list)."
        )
        return

    # Ensure the destination directory exists.
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Check whether the destination directory is the same as any source directory to avoid data loss.
    for src_file in src_files:
        src_dir = os.path.dirname(src_file)
        if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            logger.error(
                f"Cannot move files; the source directory {src_dir} is the same as the destination directory {dst_dir}."
            )
            return

    # Move the files from src to dst_dir.
    for src_file in src_files:
        if os.path.isfile(src_file):
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            shutil.move(src_file, dst_file)
        else:
            logger.warning(f"{src_file} is not a valid file path. Skipping...")

    # Optionally, delete the source directory or source directories of moved files.
    if delete_src:
        if isinstance(src, str) and os.path.exists(src):
            shutil.rmtree(src)
            logger.info(f"Deleted source directory {src}")
        elif isinstance(src, list):
            for src_file in src_files:
                src_dir = os.path.dirname(src_file)
                if os.path.exists(src_dir) and not os.listdir(src_dir):
                    os.rmdir(src_dir)
                    logger.info(f"Deleted source directory {src_dir}")


def get_all_subdirectories(directory: str) -> List[str]:
    """Return a list of all subdirectories in the given directory, including the directory itself."""
    return [dirpath for dirpath, dirnames, filenames in os.walk(directory)]


def find_parent_directory(
    path: str, directory_name: str, stop_directory: str = ""
) -> Union[str, None]:
    """
    Find the path to the parent directory that contains the specified directory name.

    Parameters:
        path (str): The path to start the search from.
        directory_name (str): The name of the directory to search for.
        stop_directory (str): Optional. A directory name to stop the search at.
                              If this is specified, the search will stop when this
                              directory is reached. If not specified, the search will
                              continue until the top-level directory is reached.

    Returns:
        str or None: The path to the parent directory containing the directory with
                     the specified name, or None if the directory is not found.
    """
    while True:
        # check if the current directory name contains the target directory name
        if directory_name in os.path.basename(path):
            return path

        # get the parent directory
        parent_dir = os.path.dirname(path)

        # check if the parent directory is the same as the current directory
        if parent_dir == path or os.path.basename(path) == stop_directory:
            print(
                f"Reached top-level directory without finding '{directory_name}':", path
            )
            return None

        # update the path to the parent directory and continue the loop
        path = parent_dir


def config_to_file(config: Union[dict, gpd.GeoDataFrame], filepath: str):
    """Saves config to config.json or config_gdf.geojson
    config's type is dict or geodataframe respectively

    Args:
        config (Union[dict, gpd.GeoDataFrame]): data to save to config file
        filepath (str): full path to directory to save config file. NOT INCLUDING THE FILE
    """
    # default save path
    filepath = str(filepath)
    save_path = filepath
    # check if config.json or config_gdf.geojson in the filepath
    if filepath.endswith("config.json"):
        filename = f"config.json"
        write_to_json(filepath, config)
    elif filepath.endswith("config_gdf.geojson"):
        filename = f"config.json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        config.to_file(filepath, driver="GeoJSON")
    elif isinstance(config, dict):
        filename = f"config.json"
        save_path = os.path.abspath(os.path.join(filepath, filename))
        write_to_json(save_path, config)
    elif isinstance(config, gpd.GeoDataFrame):
        filename = f"config_gdf.geojson"
        save_path = os.path.abspath(os.path.join(filepath, filename))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        config.to_file(save_path, driver="GeoJSON")
        
    logger.info(f"Saved {filename} saved to {save_path}")


def filter_files(files: List[str], avoid_patterns: List[str]) -> List[str]:
    """
    Filter a list of filepaths based on a list of avoid patterns.

    Args:
        files: A list of filepaths to filter.
        avoid_patterns: A list of regular expression patterns to avoid.

    Returns:
        A list of filepaths whose filenames do not match any of the patterns in avoid_patterns.

    Examples:
        >>> files = ['/path/to/file1.txt', '/path/to/file2.txt', '/path/to/avoid_file.txt']
        >>> avoid_patterns = ['.*avoid.*']
        >>> filtered_files = filter_files(files, avoid_patterns)
        >>> print(filtered_files)
        ['/path/to/file1.txt', '/path/to/file2.txt']

    """
    filtered_files = []
    for file in files:
        # Check if the file's name matches any of the avoid patterns
        for pattern in avoid_patterns:
            if re.match(pattern, os.path.basename(file)):
                break
        else:
            # If the file's name does not match any of the avoid patterns, add it to the filtered files list
            filtered_files.append(file)
    return filtered_files


def save_to_geojson_file(out_file: str, geojson: dict, **kwargs) -> None:
    """save_to_geojson_file Saves given geojson to a geojson file at outfile
    Args:
        out_file (str): The output file path
        geojson (dict): geojson dict containing FeatureCollection for all geojson objects in selected_set
    """
    # Save the geojson to a file
    out_file = check_file_path(out_file)
    ext = os.path.splitext(out_file)[1].lower()
    if ext == ".geojson":
        out_geojson = out_file
    else:
        out_geojson = os.path.splitext(out_file)[1] + ".geojson"
    with open(out_geojson, "w") as f:
        json.dump(geojson, f, **kwargs)


def check_file_path(file_path, make_dirs=True):
    """Gets the absolute file path.

    Args:
        file_path (str): The path to the file.
        make_dirs (bool, optional): Whether to create the directory if it does not exist. Defaults to True.

    Raises:
        FileNotFoundError: If the directory could not be found.
        TypeError: If the input directory path is not a string.

    Returns:
        str: The absolute path to the file.
    """
    if isinstance(file_path, str):
        if file_path.startswith("~"):
            file_path = os.path.expanduser(file_path)
        else:
            file_path = os.path.abspath(file_path)

        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir) and make_dirs:
            os.makedirs(file_dir)

        return file_path

    else:
        raise TypeError("The provided file path must be a string.")


def get_session_location(
    session_name: str = "", base_path: str = "", raise_error: bool = False
) -> str:
    """
    Gets the session location path based on the provided session name.
    If the session directory doesn't exist, it creates one.

    Parameters:
    - session_name (str, optional): Name of the session. Defaults to "".
    - base_path (str, optional): location of parent directory containing sessions directory. Defaults to "".
    - raise_error (bool,optional): Raise a FileNotFound error is session_path does not exist. Defaults to False
    Returns:
    - str: Path to the session location.

    Note:
    - This function assumes the presence of a function named `create_directory`.
    """
    if not (base_path and os.path.exists(base_path)):
        base_path = os.path.abspath(core_utilities.get_base_dir())
    session_dir = "sessions"
    session_path = (
        os.path.join(base_path, session_dir, session_name)
        if session_name
        else os.path.join(base_path, session_dir)
    )
    # if raise error is True and the session_path does not exist raise FileNotFound
    if not os.path.exists(session_path) and raise_error:
        raise FileNotFoundError(session_path)

    return session_path


def file_exists(file_path: str, filename: str) -> bool:
    """Helper function to check if a file exists and log its status."""
    if os.path.exists(file_path):
        logger.info(f"{filename} exists at location: {file_path}")
        return True

    logger.warning(f"{filename} file missing at {file_path}")
    return False


def extract_roi_id(path: str) -> str:
    """extracts the ROI ID from the path

    Args:
        path (str): path containing ROI directory

    Returns:
        str: ID of the ROI within the path
    """
    pattern = r"ID_([A-Za-z0-9]+)"
    match = re.search(pattern, path)
    if match:
        return match.group(1)
    else:
        return None


def find_matching_directory_by_id(base_directory: str, roi_id: str) -> str:
    """
    Loops through all directories in the given base_directory to find a directory with a matching ROI ID.

    Args:
        base_directory (str): Path to the directory containing subdirectories to check.
        roi_id (str): ROI ID to match against directories.

    Returns:
        str: Path to the matching directory if found. Otherwise, returns None.
    """

    # Iterate over each directory in the base_directory
    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)

        # Check if it's a directory and has the matching ROI ID
        if os.path.isdir(dir_path) and extract_roi_id(dir_name) == roi_id:
            return dir_path

    # If no matching directory is found, return None
    return None


def create_session_path(session_name: str, ROI_directory_name: str) -> str:
    """
    Creates a session path by joining the current working directory, a fixed "sessions" directory, and the provided session name.
    After constructing the path, it further creates the sub directory provided ROI_directory.

    Parameters:
    - session_name (str): Name of the session for which the path has to be created.
    - ROI_directory_name (str): The name of the directory related to the region of interest (ROI) that will be appended to the session path.

    Returns:
    - str: The path to the newly created session directory.

    Note:
    - This function assumes the presence of a function named `create_directory` and a logger object named `logger`.
    """    
    base_dir = os.path.abspath(core_utilities.get_base_dir())
    session_path = os.path.join(base_dir, "sessions", session_name)
    session_path = create_directory(session_path, ROI_directory_name)
    logger.info(f"session_path: {session_path}")
    return session_path


def create_directory(file_path: str, name: str) -> str:
    """Creates a new directory with the given name at the specified file path.

    Args:
        file_path (str): The file path where the new directory will be created.
        name (str): The name of the new directory to be created.

    Returns:
        str: The full path of the new directory created or existing directory if it already existed.

    Raises:
        OSError: If there was an error creating the new directory.
    """
    # if file_path is not a path then convert it to a path
    if not isinstance(file_path, os.PathLike):
        file_path = os.path.abspath(file_path)
    new_directory = os.path.join(file_path, name)
    # If the directory named 'name' does not exist, create it
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    return new_directory


def write_to_json(filepath: str, settings: dict):
    """ "Write the  settings dictionary to json file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    to_file(settings, filepath)


def to_file(data: dict, filepath: str) -> None:
    """
    Serializes a dictionary to a JSON file, handling special serialization for datetime and numpy ndarray objects.

    The function handles two special cases:
    1. If the data contains datetime.date or datetime.datetime objects, they are serialized to their ISO format.
    2. If the data contains numpy ndarray objects, they are converted to lists before serialization.

    Parameters:
    - data (dict): Dictionary containing the data to be serialized to a JSON file.
    - filepath (str): Path (including filename) where the JSON file should be saved.

    Returns:
    - None

    Note:
    - This function requires the json, datetime, and numpy modules to be imported.
    """

    class DateTimeEncoder(JSONEncoder):
        # Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            # Check for numpy arrays
            if isinstance(obj, np.ndarray):
                # Check if the dtype is 'object', which indicates it might have mixed types including datetimes
                if obj.dtype == "object":
                    # Convert each element of the array
                    return [self.default(item) for item in obj]
                else:
                    # If it's not an object dtype, simply return the array as a list
                    return obj.tolist()

    with open(filepath, "w") as fp:
        json.dump(data, fp, cls=DateTimeEncoder)


def copy_files_to_dst(src_path: str, dst_path: str, glob_str: str) -> None:
    """Copies all files from src_path to dest_path
    Args:
        src_path (str): full path to the data directory in coastseg
        dst_path (str): full path to the images directory in Sniffer
    """
    if not os.path.exists(dst_path):
        raise FileNotFoundError(f"dst_path: {dst_path} doesn't exist.")
    elif not os.path.exists(src_path):
        raise FileNotFoundError(f"src_path: {src_path} doesn't exist.")
    else:
        for file in glob.glob(glob_str):
            shutil.copy(file, dst_path)
        logger.info(f"\nCopied files that matched {glob_str}  \nto {dst_path}")


def find_directory_recursively(path: str = ".", name: str = "RGB") -> str:
    """
    Recursively search for a directory named "RGB" in the given path or its subdirectories.

    Args:
        path (str): The starting directory to search in. Defaults to current directory.

    Returns:
        str: The path of the first directory named "RGB" found, or None if not found.
    """
    dir_location = None
    if os.path.basename(path) == name:
        dir_location = path
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            if name in dirnames:
                dir_location = os.path.join(dirpath, name)

    if not os.listdir(dir_location):
        raise Exception(f"{name} directory is empty.")

    if not dir_location:
        raise Exception(f"No directroy matching {name} could be found at {path}")

    return dir_location


def find_files_recursively(
    path: str = ".", search_pattern: str = "*RGB*", raise_error: bool = False
) -> List[str]:
    """
    Recursively search for files with the given search pattern in the given path or its subdirectories.

    Args:
        path (str): The starting directory to search in. Defaults to current directory.
        search_pattern (str): The search pattern to match against file names. Defaults to "*RGB*".
        raise_error (bool): Whether to raise an error if no files are found. Defaults to False.

    Returns:
        list: A list of paths to all files that match the given search pattern.
    """
    file_locations = []
    regex = re.compile(search_pattern, re.IGNORECASE)
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if regex.match(filename):
                file_location = os.path.join(dirpath, filename)
                file_locations.append(file_location)

    if not file_locations and raise_error:
        raise Exception(f"No files matching {search_pattern} could be found at {path}")

    return file_locations

def find_files_in_directory(
    path: str = ".", search_pattern: str = "*RGB*", raise_error: bool = False
) -> List[str]:
    """
    Return a list of files with the given search pattern in the given directory.

    Args:
        path (str): The starting directory to search in. Defaults to current directory.
        search_pattern (str): The search pattern to match against file names. Defaults to "*RGB*".
        raise_error (bool): Whether to raise an error if no files are found. Defaults to False.

    Returns:
        list: A list of paths to all files that match the given search pattern.
    """
    file_locations = []
    regex = re.compile(search_pattern, re.IGNORECASE)
    filenames = os.listdir(path)
    for filename in filenames:
        if regex.match(filename):
            file_location = os.path.join(path, filename)
            file_locations.append(file_location)

    if not file_locations and raise_error:
        raise Exception(f"No files matching {search_pattern} could be found at {path}")

    return file_locations

def find_file_recursively(path: str = ".", name: str = "RGB") -> str:
    """
    Recursively search for a file named "RGB" in the given path or its subdirectories.

    Args:
        path (str): The starting directory to search in. Defaults to current directory.

    Returns:
        str: The path of the first directory named "RGB" found, or None if not found.
    """
    file_location = None
    if os.path.basename(path) == name:
        file_location = path
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            if name in filenames:
                file_location = os.path.join(dirpath, name)
                return file_location

    if not os.listdir(file_location):
        raise Exception(f"{file_location} is empty.")

    if not file_location:
        raise Exception(f" No file matching {name} could be found at {path}")

    return file_location


def mk_new_dir(name: str, location: str):
    """Create new folder with name_datetime stamp at location
    Args:
        name (str): name of folder to create
        location (str): full path to location to create folder
    """
    if os.path.exists(location):
        new_folder = location + os.sep + name + "_" + generate_datestring()
        os.mkdir(new_folder)
        return new_folder
    else:
        raise Exception("Location provided does not exist.")


def read_geojson_file(geojson_file: str) -> dict:
    """Returns the geojson of the selected ROIs from the file specified by geojson_file"""
    with open(geojson_file) as f:
        data = geojson.load(f)
    return data


def read_gpd_file(filename: str) -> gpd.GeoDataFrame:
    """
    Returns geodataframe from geopandas geodataframe file
    """
    if os.path.exists(filename):
        logger.info(f"Opening \n {filename}")
        return gpd.read_file(filename)
    else:
        raise FileNotFoundError


def load_json_data_from_file(search_location: str, filename: str) -> dict:
    """
    Load JSON data from a file by searching for the file in the specified location.

    The function searches recursively in the provided search location for a file with
    the specified filename. Once the file is found, it loads the JSON data from the file
    and returns it as a dictionary.

    Args:
        search_location (str): Directory or path to search for the file.
        filename (str): Name of the file to load.

    Returns:
        dict: Data read from the JSON file as a dictionary.

    """
    file_path = find_file_recursively(search_location, filename)
    json_data = load_data_from_json(file_path)
    return json_data


def load_data_from_json(filepath: str) -> dict:
    """
    Reads data from a JSON file and returns it as a dictionary.

    The function reads the data from the specified JSON file using the provided filepath.
    It applies a custom object hook, `DecodeDateTime`, to decode the datetime and shoreline
    data if they exist in the dictionary.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Data read from the JSON file as a dictionary.

    """

    def DecodeDateTime(readDict):
        """
        Helper function to decode datetime and shoreline data in the dictionary.

        Args:
            readDict (dict): Dictionary to decode.

        Returns:
            dict: Decoded dictionary.

        """
        if "dates" in readDict:
            tmp = [
                datetime.datetime.fromisoformat(dates) for dates in readDict["dates"]
            ]
            readDict["dates"] = tmp
        if "shorelines" in readDict:
            tmp = [
                np.array(shoreline) if len(shoreline) > 0 else np.empty((0, 2))
                for shoreline in readDict["shorelines"]
            ]
            readDict["shorelines"] = tmp
        return readDict

    with open(filepath, "r") as fp:
        data = json.load(fp, object_hook=DecodeDateTime)
    return data


def rename_jpgs(src_path: str) -> None:
    """Renames all the jpgs in the data directory in coastseg
    Args:
        src_path (str): full path to the data directory in coastseg
    """
    files_renamed = False
    for folder in os.listdir(src_path):
        folder_path = src_path + os.sep + folder
        # Split the folder name at the first _
        folder_id = folder.split("_")[0]
        folder_path = folder_path + os.sep + "jpg_files" + os.sep + "preprocessed"
        jpgs = glob.glob1(folder_path + os.sep, "*jpg")
        # Append folder id to basename of jpg if not already there
        for jpg in jpgs:
            if folder_id not in jpg:
                files_renamed = True
                base, ext = os.path.splitext(jpg)
                new_name = folder_path + os.sep + base + "_" + folder_id + ext
                old_name = folder_path + os.sep + jpg
                os.rename(old_name, new_name)
        if files_renamed:
            print(f"Renamed files in {src_path} ")
