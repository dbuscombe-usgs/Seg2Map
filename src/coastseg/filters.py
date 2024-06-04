import logging
from statistics import mode
import numpy as np
import xarray as xr
import os, shutil
from sklearn.cluster import KMeans
from statistics import mode
import pathlib

# Logger setup
logger = logging.getLogger(__name__)


def copy_files(files: list, dest_folder: str) -> None:
    """
    Copy files to a specified destination folder.

    Args:
        files (list): List of file paths to be copied.
        dest_folder (str): Destination folder where files will be copied.

    Returns:
        None
    """
    os.makedirs(dest_folder, exist_ok=True)
    for f in files:
        dest_path = os.path.join(dest_folder, os.path.basename(f))
        if os.path.exists(os.path.abspath(dest_path)):
            continue
        # shutil.copy(f, dest_folder)
        shutil.move(f, dest_path)
        # move the matching png files to the respective folders
        parent_dir = os.path.abspath(pathlib.Path(f).parent)
        png_file = os.path.basename(f.replace("_res.npz","_predseg.png"))
        png_path = os.path.join(parent_dir, png_file)
        if os.path.exists(png_path):
            dest_path = os.path.join(dest_folder, os.path.basename(png_path))
            shutil.move(png_path, dest_path)
            


def load_data(f: str) -> np.array:
    """
    Load the data from the specified .npz file and extract the 'grey_label' array.

    Args:
        f (str): Path to the .npz file.

    Returns:
        np.array: The 'grey_label' array from the .npz file.
    """
    with np.load(f) as data:
        grey = data["grey_label"].astype("uint8")
    return grey


def get_good_bad_files(files: list, labels: np.array, scores: list) -> tuple:
    """
    Split files into 'good' and 'bad' categories based on provided labels and scores.

    Args:
        files (list): List of file paths.
        labels (np.array): Array of labels corresponding to the files.
        scores (list): List of scores associated with the files.

    Returns:
        tuple: A tuple containing two arrays:
            - files_bad (np.array): Array of 'bad' categorized file paths (highest score label).
            - files_good (np.array): Array of 'good' categorized file paths (lowest score label).
    """
    # drop NaN values from scores
    scores = np.array(scores)[~np.isnan(scores)]
    # if all the scores or labels are the same return all the files as 'good' even if they may be bad
    if len(np.unique(scores)) == 1 or len(np.unique(labels)) == 1:
        return [], np.array(files)
    files_bad = np.array(files)[labels == np.argmax(scores)]
    files_good = np.array(files)[labels == np.argmin(scores)]
    return files_bad, files_good


def get_time_vectors(files: list) -> tuple:
    """
    Extract time information from a list of file paths and create an xarray variable.

    Args:
        files (list): List of file paths containing time information.

    Returns:
        tuple: A tuple containing two elements:
            - times (list): List of time values extracted from the file paths.
            - time_variable (xr.Variable): xarray variable containing the time values.
    """
    times = [f.split(os.sep)[-1].split("_")[0] for f in files]
    return times, xr.Variable("time", times)


def get_image_shapes(files: list) -> list:
    return [load_data(f).shape for f in files]


def measure_rmse(da: xr.DataArray, times: list, timeav: xr.DataArray) -> tuple:
    """
    Measure the Root Mean Square Error (RMSE) between data arrays and their average.

    Args:
        da (xr.DataArray): Data array containing time series data.
        times (list): List of time values.
        timeav (xr.DataArray): Time-averaged data array.

    Returns:
        tuple: List of RMSE values and their reshaped version.
    """
    rmse = [
        float(np.sqrt(np.mean((da.sel(time=t) - timeav) ** 2)).to_numpy())
        for t in times
    ]
    input_rmse = np.array(rmse).reshape(-1, 1)
    return rmse, input_rmse


def get_kmeans_clusters(input_rmse: np.array, rmse: list) -> tuple:
    """
    Perform KMeans clustering on RMSE values. 
    Returns the average rmse score for each cluster as well as the labels for each cluster.
    score[0] is the average rmse for the cluster 0.
    score [1] is the average rmse for the cluster 1.

    Args:
        input_rmse (np.array): Array of RMSE values.
        rmse (list): List of RMSE values.

    Returns:
        tuple: Labels assigned by KMeans and mean score for each cluster.
    """
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(input_rmse)
    labels = kmeans.labels_
    # Calculate mean RMSE for each cluster
    # the lower the RMSE the better the prediction
    scores = [
        np.mean(np.array(rmse)[labels == 0]),
        np.mean(np.array(rmse)[labels == 1]),
    ]
    return labels, scores


def load_xarray_data(f: str) -> xr.DataArray:
    """
    Load data from a specified .npz file and convert it to an xarray DataArray.

    Args:
        f (str): Path to the .npz file.

    Returns:
        xr.DataArray: Data loaded from file as an xarray DataArray.
    """
    with np.load(f) as data:
        grey = data["grey_label"].astype("uint8")
    ny, nx = grey.shape
    y = np.arange(ny)
    x = np.arange(nx)
    return xr.DataArray(grey, coords={"y": y, "x": x}, dims=["y", "x"])


def handle_files_and_directories(
    files_bad: list, files_good: list, dest_folder_bad: str, dest_folder_good: str
) -> None:
    """
    Organize files into 'good' and 'bad' directories by copying them to respective folders.

    Args:
        files_bad (list): List of file paths categorized as 'bad'.
        files_good (list): List of file paths categorized as 'good'.
        dest_folder_bad (str): Destination folder for 'bad' files.
        dest_folder_good (str): Destination folder for 'good' files.
    """
    os.makedirs(dest_folder_bad, exist_ok=True)
    os.makedirs(dest_folder_good, exist_ok=True)
    logger.info(f"Copying {len(files_bad)} files to {dest_folder_bad}")
    logger.info(f"Copying {len(files_good)} files to {dest_folder_good}")
    copy_files(files_bad, dest_folder_bad)
    copy_files(files_good, dest_folder_good)
    # get the matching png files and copy them to the respective folders


def return_valid_files(files: list) -> list:
    """
    Return files whose image shapes match the most common shape among the given files.

    Args:
        files (list): List of file paths.

    Returns:
        list: File paths whose image shape matches the mode of all file shapes.
    """
    modal_shape = mode(get_image_shapes(files))
    return [f for f in files if load_data(f).shape == modal_shape]

def apply_land_mask( directory_path: str) -> None:
    """
    Apply land mask to 'good' files and save them to the destination folder.

    Args:
        dest_folder_good (str): Destination folder for 'good' files
    """
    files_good = [os.path.join(directory_path, file) for file in os.listdir(directory_path)]
    if len(files_good) == 0:
        return directory_path
    times, time_var = get_time_vectors(files_good)
    # create xarray from good files
    da = xr.concat([load_xarray_data(f) for f in files_good], dim=time_var)
    # create time average of good files
    timeav = da.mean(dim="time")
    # create land mask from the time averaged image
    mask_land = np.array(np.round(timeav)==3).astype('int')
    # apply land mask to each time in the good files
    for time in times:
        # select the time
        frame = da.sel(time=time).to_numpy()
        frame[mask_land==1] = 3
        da.sel(time=time).values = frame

    # save the masked files to npz
    for f in files_good:
        dest_path = os.path.join(directory_path, os.path.basename(f))
        if not os.path.exists(os.path.dirname(dest_path)):
            os.makedirs(os.path.dirname(dest_path),exist_ok=True)
        np.savez_compressed(dest_path, grey_label=da.sel(time=time).to_numpy())
        
    return directory_path



def filter_model_outputs(
    satname: str, files: list, dest_folder_good: str, dest_folder_bad: str
) -> None:
    """
    Filter model outputs based on KMeans clustering of RMSE values and organize into 'good' and 'bad'.

    Args:
        label (str): Label used for categorizing.
        files (list): List of file paths.
        dest_folder_good (str): Destination folder for 'good' files.
        dest_folder_bad (str): Destination folder for 'bad' files.
    """
    valid_files = return_valid_files(files)
    if len(valid_files) <3:
        # if there are not enough valid files to perform the analysis, move all files to the good folder
        handle_files_and_directories(
        [], valid_files, dest_folder_bad, dest_folder_good
        ) 
        return 
    print(f"Found {len(valid_files)} valid files for {satname}.")
    times, time_var = get_time_vectors(valid_files)
    da = xr.concat([load_xarray_data(f) for f in valid_files], dim=time_var)
    timeav = da.mean(dim="time")

    rmse, input_rmse = measure_rmse(da, times, timeav)
    labels, scores = get_kmeans_clusters(input_rmse, rmse)
    files_bad, files_good = get_good_bad_files(valid_files, labels, scores)
    handle_files_and_directories(
        files_bad, files_good, dest_folder_bad, dest_folder_good
    ) 
    
    
    
