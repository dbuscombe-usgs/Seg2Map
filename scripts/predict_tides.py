# Standard library imports
import os
import pathlib
from pathlib import Path
import time
import argparse
from typing import Dict, List, Tuple, Union

# Third-party imports
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pyTMD.io
import pyTMD.io.model
import pyTMD.predict
import pyTMD.spatial
import pyTMD.time
import pyTMD.utilities
from shapely.geometry import Point
import shapely
from shapely.geometry import LineString

# Instructions for running the script
# -----------------------------------
# Example 1: 
# Run the script using a single point in the GeoJSON file parranporth.geojson,
# the time series data in the CSV file transect_time_series.csv,
# and the FES 2014 model region is located in the directory tide_model.
# python predict_tides.py -C "C:\Users\sf230\Downloads\parranporth.geojson" -T "C:\development\doodleverse\coastsat_package\coastsat_package\data\NARRA\transect_time_series.csv"  -M "C:\development\doodleverse\coastseg\CoastSeg\tide_model"

# the output will be saved in the current working directory as tidal_predictions.csv
# the output file will have the columns dates,x,y,tide, where x and y are the coordinates of the point in the GeoJSON file
# -----------------------------------

def convert_date_gdf(gdf):
    """
    Convert date columns in a GeoDataFrame to datetime format.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame.

    Returns:
        GeoDataFrame: The converted GeoDataFrame with date columns in datetime format.
    """
    gdf = gdf.copy()
    if 'dates' in gdf.columns:
        gdf['dates'] = pd.to_datetime(gdf['dates']).dt.tz_convert(None)
    if 'date' in gdf.columns:
        gdf['date'] = pd.to_datetime(gdf['date']).dt.tz_convert(None)
    gdf = stringify_datetime_columns(gdf)
    return gdf

def create_complete_line_string(points):
    """
    Create a complete LineString from a list of points.
    If there is only a single point in the list, a Point object is returned instead of a LineString.

    Args:
        points (numpy.ndarray): An array of points representing the coordinates.

    Returns:
        LineString: A LineString object representing the complete line.

    Raises:
        None.

    """
    # Ensure all points are unique to avoid redundant looping
    unique_points = np.unique(points, axis=0)
    
    # Start with the first point in the list
    if len(unique_points) == 0:
        return None  # Return None if there are no points
    
    starting_point = unique_points[0]
    current_point = starting_point
    sorted_points = [starting_point]
    visited_points = {tuple(starting_point)}

    # Repeat until all points are visited
    while len(visited_points) < len(unique_points):
        nearest_distance = np.inf
        nearest_point = None
        for point in unique_points:
            if tuple(point) in visited_points:
                continue  # Skip already visited points
            # Calculate the distance to the current point
            distance = np.linalg.norm(point - current_point)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_point = point

        # Break if no unvisited nearest point was found (should not happen if all points are unique)
        if nearest_point is None:
            break

        sorted_points.append(nearest_point)
        visited_points.add(tuple(nearest_point))
        current_point = nearest_point

    # Convert the sorted list of points to a LineString
    if len(sorted_points) < 2:
        return Point(sorted_points[0])

    return LineString(sorted_points)


def order_linestrings_gdf(gdf,dates, output_crs='epsg:4326'):
    """
    Orders the linestrings in a GeoDataFrame by creating complete line strings from the given points.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame containing linestrings.
        dates (list): The list of dates corresponding to the linestrings.
        output_crs (str): The output coordinate reference system (CRS) for the GeoDataFrame. Default is 'epsg:4326'.

    Returns:
        GeoDataFrame: The ordered GeoDataFrame with linestrings.

    """
    gdf = gdf.copy()
    # Convert to the output CRS
    if gdf.crs is not None:
        gdf.to_crs(output_crs, inplace=True)
    else:
        gdf.set_crs(output_crs, inplace=True)
        
    all_points = [shapely.get_coordinates(p) for p in gdf.geometry]
    lines = []
    for points in all_points:
        line_string = create_complete_line_string(points)
        lines.append(line_string)
    gdf = gpd.GeoDataFrame({'geometry': lines,'date': dates},crs=output_crs)
    return gdf

def convert_points_to_linestrings(gdf, group_col='date', output_crs='epsg:4326') -> gpd.GeoDataFrame:
    """
    Convert points to LineStrings.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame containing points.
        group_col (str): The column to group the GeoDataFrame by (default is 'date').
        output_crs (str): The coordinate reference system for the output GeoDataFrame (default is 'epsg:4326').

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame containing LineStrings created from the points.
    """
    # Group the GeoDataFrame by date
    grouped = gdf.groupby(group_col)
    # For each group, ensure there are at least two points so that a LineString can be created
    filtered_groups = grouped.filter(lambda g: g[group_col].count() > 1)
    # Recreate the groups as a geodataframe
    grouped_gdf = gpd.GeoDataFrame(filtered_groups, geometry='geometry')
    linestrings = grouped_gdf.groupby(group_col).apply(lambda g: LineString(g.geometry.tolist()))

    # Create a new GeoDataFrame from the LineStrings
    linestrings_gdf = gpd.GeoDataFrame(linestrings, columns=['geometry'], crs=output_crs)
    linestrings_gdf.reset_index(inplace=True)
    
    # order the linestrings so that they are continuous
    linestrings_gdf = order_linestrings_gdf(linestrings_gdf,linestrings_gdf['date'],output_crs=output_crs)
    
    return linestrings_gdf


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


def export_dataframe_as_geojson(data:pd.DataFrame, output_file_path:str, x_col:str, y_col:str, id_col:str,columns_to_keep:List[str] = None)->str:
    """
    Export specified columns from a CSV file to a GeoJSON format, labeled by a unique identifier.
    
    Parameters:
    - data: pd.DataFrame, the input data.
    - output_file_path: str, path for the output GeoJSON file.
    - x_col: str, column name for the x coordinates (longitude).
    - y_col: str, column name for the y coordinates (latitude).
    - id_col: str, column name for the unique identifier (transect id).
    - columns_to_keep: List[str], list of columns to keep in the output GeoJSON file. Defaults to None.
    
    Returns:
    - str, path for the created GeoJSON file.
    """

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data, 
        geometry=[Point(xy) for xy in zip(data[x_col], data[y_col])], 
        crs="EPSG:4326"
    )
    
    if columns_to_keep:
        columns_to_keep.append(id_col)
        columns_to_keep.append('geometry')
        gdf = gdf[columns_to_keep].copy()
        if 'dates' in gdf.columns:
            gdf['dates'] = pd.to_datetime(gdf['dates']).dt.tz_convert(None)
        if 'date' in gdf.columns:
            gdf['date'] = pd.to_datetime(gdf['date']).dt.tz_convert(None)
        gdf = stringify_datetime_columns(gdf)
    else:
        # Keep only necessary columns
        gdf = gdf[[id_col, 'geometry']].copy()
        
    
    # Export to GeoJSON
    gdf.to_file(output_file_path, driver='GeoJSON')
    
    # Return the path to the output file
    return output_file_path

def get_tide_predictions(
   x:float,y:float, timeseries_df: pd.DataFrame, model_region_directory: str,transect_id:str="",
) -> pd.DataFrame:
    """
    Get tide predictions for a given location and transect ID.

    Args:
        x (float): The x-coordinate of the location to predict tide for.
        y (float): The y-coordinate of the location to predict tide for.
        - timeseries_df: A DataFrame containing time series data for each transect.
       - model_region_directory: The path to the FES 2014 model region that will be used to compute the tide predictions
         ex."C:/development/doodleverse/CoastSeg/tide_model/region"
        transect_id (str): The ID of the transect. Pass "" if no transect ID is available.
        
    Returns:
            - pd.DataFrame: A DataFrame containing tide predictions for all the dates that the selected transect_id using the
    fes 2014 model region specified in the "region_id".
    """
    print(f"Loading the tide model from: {model_region_directory}")
    # if the transect ID is not in the timeseries_df then return None
    if transect_id != "":
        if transect_id not in timeseries_df.columns:
            return None
        dates_for_transect_id_df = timeseries_df[["dates", transect_id]].dropna()
    if transect_id == "":
        dates_for_transect_id_df = timeseries_df[["dates"]].dropna()
    tide_predictions_df = model_tides(
        x,
        y,
        dates_for_transect_id_df.dates.values,
        directory=model_region_directory,
    )
    return tide_predictions_df


def predict_tides_for_df(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Predict tides for a points in the DataFrame.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward points for each transect
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing time series data for the transects
    - config: Configuration dictionary.
        Must contain key:
        "REGION_DIRECTORY" : contains full path to the fes 2014 model region folder

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides.
    Contains columns dates,x,y,tide,transect_id
    """
    region_directory = config["REGION_DIRECTORY"]
    # Apply the get_tide_predictions over each row and collect results in a list
    all_tides = seaward_points_gdf.apply(
        lambda row: get_tide_predictions(row.geometry.x,
                                         row.geometry.y,
                                         timeseries_df,
                                         f"{region_directory}{row['region_id']}"),axis=1)
    # Filter out None values
    all_tides = all_tides.dropna()
    # if no tides are predicted return an empty dataframe
    if all_tides.empty:
        return pd.DataFrame(columns=["dates", "x", "y", "tide",])
    
    # Concatenate all the results
    all_tides_df = pd.concat(all_tides.tolist())

    return all_tides_df

def model_tides(
    x,
    y,
    time,
    transect_id: str = "",
    model="FES2014",
    directory=None,
    epsg=4326,
    method="bilinear",
    extrapolate=True,
    cutoff=10.0,
):
    """
    Compute tides at points and times using tidal harmonics.
    If multiple x, y points are provided, tides will be
    computed for all timesteps at each point.

    This function supports any tidal model supported by
    `pyTMD`, including the FES2014 Finite Element Solution
    tide model, and the TPXO8-atlas and TPXO9-atlas-v5
    TOPEX/POSEIDON global tide models.

    This function requires access to tide model data files
    to work. These should be placed in a folder with
    subfolders matching the formats specified by `pyTMD`:
    https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories

    For FES2014 (https://www.aviso.altimetry.fr/es/data/products/auxiliary-products/global-tide-fes/description-fes2014.html):
        - {directory}/fes2014/ocean_tide/
          {directory}/fes2014/load_tide/

    For TPXO8-atlas (https://www.tpxo.net/tpxo-products-and-registration):
        - {directory}/tpxo8_atlas/

    For TPXO9-atlas-v5 (https://www.tpxo.net/tpxo-products-and-registration):
        - {directory}/TPXO9_atlas_v5/

    This function is a minor modification of the `pyTMD`
    package's `compute_tide_corrections` function, adapted
    to process multiple timesteps for multiple input point
    locations. For more info:
    https://pytmd.readthedocs.io/en/stable/user_guide/compute_tide_corrections.html

    Parameters:
    -----------
    x, y : float or list of floats
        One or more x and y coordinates used to define
        the location at which to model tides. By default these
        coordinates should be lat/lon; use `epsg` if they
        are in a custom coordinate reference system.
    time : A datetime array or pandas.DatetimeIndex
        An array containing 'datetime64[ns]' values or a
        'pandas.DatetimeIndex' providing the times at which to
        model tides in UTC time.
    model : string
        The tide model used to model tides. Options include:
        - "FES2014" (only pre-configured option on DEA Sandbox)
        - "TPXO8-atlas"
        - "TPXO9-atlas-v5"
    directory : string
        The directory containing tide model data files. These
        data files should be stored in sub-folders for each
        model that match the structure provided by `pyTMD`:
        https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories
        For example:
        - {directory}/fes2014/ocean_tide/
          {directory}/fes2014/load_tide/
        - {directory}/tpxo8_atlas/
        - {directory}/TPXO9_atlas_v5/
    epsg : int
        Input coordinate system for 'x' and 'y' coordinates.
        Defaults to 4326 (WGS84).
    method : string
        Method used to interpolate tidal contsituents
        from model files. Options include:
        - bilinear: quick bilinear interpolation
        - spline: scipy bivariate spline interpolation
        - linear, nearest: scipy regular grid interpolations
    extrapolate : bool
        Whether to extrapolate tides for locations outside of
        the tide modelling domain using nearest-neighbor
    cutoff : int or float
        Extrapolation cutoff in kilometers. Set to `np.inf`
        to extrapolate for all points.

    Returns
    -------
    A pandas.DataFrame containing tide heights for all the xy points and their corresponding time
    """
    # Check tide directory is accessible
    if directory is not None:
        directory = pathlib.Path(directory).expanduser()
        if not directory.exists():
            raise FileNotFoundError("Invalid tide directory")

    # Validate input arguments
    assert method in ("bilinear", "spline", "linear", "nearest")

    # Get parameters for tide model; use custom definition file for
    model = pyTMD.io.model(directory, format="netcdf", compressed=False).elevation(
        model
    )

    # If time passed as a single Timestamp, convert to datetime64
    if isinstance(time, pd.Timestamp):
        time = time.to_datetime64()

    # Handle numeric or array inputs
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    time = np.atleast_1d(time)

    # Determine point and time counts
    assert len(x) == len(y), "x and y must be the same length"
    n_points = len(x)
    n_times = len(time)

    # Converting x,y from EPSG to latitude/longitude
    try:
        # EPSG projection code string or int
        crs1 = pyproj.CRS.from_epsg(int(epsg))
    except (ValueError, pyproj.exceptions.CRSError):
        # Projection SRS string
        crs1 = pyproj.CRS.from_string(epsg)

    # Output coordinate reference system
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    lon, lat = transformer.transform(x.flatten(), y.flatten())

    # Convert datetime
    timescale = pyTMD.time.timescale().from_datetime(time.flatten())
    n_points = len(x)
    # number of time points
    n_times = len(time)

    if model.format == "FES":
        amp, ph = pyTMD.io.FES.extract_constants(
            lon,
            lat,
            model.model_file,
            type=model.type,
            version=model.version,
            method=method,
            extrapolate=extrapolate,
            cutoff=cutoff,
            scale=model.scale,
            compressed=model.compressed,
        )

        # Available model constituents
        c = model.constituents

        # Delta time (TT - UT1)
        # calculating the difference between Terrestrial Time (TT) and UT1 (Universal Time 1),
        deltat = timescale.tt_ut1

    # Calculate complex phase in radians for Euler's
    cph = -1j * ph * np.pi / 180.0

    # Calculate constituent oscillation
    hc = amp * np.exp(cph)

    # Repeat constituents to length of time and number of input
    # coords before passing to `predict_tide_drift`

    # deltat likely represents the time interval between successive data points or time instances.
    # t =  replicating the timescale.tide array n_points times
    # hc = creates an array with the tidal constituents repeated for each time instance
    # Repeat constituents to length of time and number of input
    # coords before passing to `predict_tide_drift`
    t, hc, deltat = (
        np.tile(timescale.tide, n_points),
        hc.repeat(n_times, axis=0),
        np.tile(deltat, n_points),
    )

    # Predict tidal elevations at time and infer minor corrections
    npts = len(t)
    tide = np.ma.zeros((npts), fill_value=np.nan)
    tide.mask = np.any(hc.mask, axis=1)

    # Predict tides
    tide.data[:] = pyTMD.predict.drift(
        t, hc, c, deltat=deltat, corrections=model.format
    )
    minor = pyTMD.predict.infer_minor(t, hc, c, deltat=deltat, corrections=model.format)
    tide.data[:] += minor.data[:]

    # Replace invalid values with fill value
    tide.data[tide.mask] = tide.fill_value
    if transect_id:
        df = pd.DataFrame(
            {
                "dates": np.tile(time, n_points),
                "x": np.repeat(x, n_times),
                "y": np.repeat(y, n_times),
                "tide": tide,
                "transect_id": transect_id,
            }
        )
        df["dates"] = pd.to_datetime(df["dates"], utc=True)
        df.set_index("dates")
        return df
    else:
        df = pd.DataFrame(
            {
                "dates": np.tile(time, n_points),
                "x": np.repeat(x, n_times),
                "y": np.repeat(y, n_times),
                "tide": tide,
            }
        )
        df["dates"] = pd.to_datetime(df["dates"], utc=True)
        df.set_index("dates")
        return df


def get_seaward_points_gdf(transects_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame containing the seaward points from a given GeoDataFrame containing transects.

    Parameters:
    - transects_gdf: A GeoDataFrame containing transect data.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the seaward points for all of the transects.
    Contains columns transect_id and geometry in crs 4326
    """
    # Set transects crs to epsg:4326 if it is not already. Tide model requires crs 4326
    if transects_gdf.crs is None:
        transects_gdf = transects_gdf.set_crs("epsg:4326")
    else:
        transects_gdf = transects_gdf.to_crs("epsg:4326")

    # Prepare data for the new GeoDataFrame
    data = []
    for index, row in transects_gdf.iterrows():
        points = list(row["geometry"].coords)
        seaward_point = Point(points[1]) if len(points) > 1 else Point()

        # Append data for each transect to the data list
        data.append({"transect_id": row["id"], "geometry": seaward_point})

    # Create the new GeoDataFrame
    seaward_points_gdf = gpd.GeoDataFrame(data, crs="epsg:4326")

    return seaward_points_gdf


def read_and_filter_geojson(
    file_path: str,
    columns_to_keep: Tuple[str, ...] = ("id", "type", "geometry"),

) -> gpd.GeoDataFrame:
    """
    Read and filter a GeoJSON file based on specified columns and feature type.

    Parameters:
    - file_path: Path to the GeoJSON file.
    - columns_to_keep: A tuple containing column names to be retained in the resulting GeoDataFrame. Default is ("id", "type", "geometry").
    - feature_type: Type of feature to be retained in the resulting GeoDataFrame. Default is "transect".

    Returns:
    - gpd.GeoDataFrame: A filtered GeoDataFrame.
    """
    # Read the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(file_path)
    # Drop all other columns in place
    gdf.drop(
        columns=[col for col in gdf.columns if col not in columns_to_keep], inplace=True
    )
    return gdf


def load_regions_from_geojson(geojson_path: str) -> gpd.GeoDataFrame:
    """
    Load regions from a GeoJSON file and assign a region_id based on index.

    Parameters:
    - geojson_path: Path to the GeoJSON file containing regions.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the loaded regions with an added 'region_id' column.
    """
    print(f"Loading regions from GeoJSON file: {geojson_path}")
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    gdf["region_id"] = gdf.index
    return gdf


def perform_spatial_join(
    seaward_points_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join between seaward points and regions based on intersection.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward points.
    - regions_gdf: A GeoDataFrame containing regions.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame resulting from the spatial join.
    """
    joined_gdf = gpd.sjoin(
        seaward_points_gdf, regions_gdf, how="left", predicate="intersects"
    )
    joined_gdf.drop(columns="index_right", inplace=True)
    return joined_gdf


def model_tides_for_all(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Model tides for all points in the provided GeoDataFrame.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward point for each transect.
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing time series data for tides.
    - config: Configuration dictionary.

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides for all points.
    """
    return predict_tides_for_df(seaward_points_gdf, timeseries_df, config)


def model_tides_by_region_id(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Model tides for each unique region ID in the provided GeoDataFrame.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward point for each transect.
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing time series data for tides.
    - config: Configuration dictionary.

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides segmented by region ID.
    """
    unique_ids = seaward_points_gdf["region_id"].unique()
    all_tides_dfs = []

    for uid in unique_ids:
        subset_gdf = seaward_points_gdf[seaward_points_gdf["region_id"] == uid]
        tides_for_region_df = predict_tides_for_df(subset_gdf, timeseries_df, config)
        all_tides_dfs.append(tides_for_region_df)

    return pd.concat(all_tides_dfs)


def handle_tide_predictions(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Handle tide predictions based on the number of unique region IDs in the provided GeoDataFrame.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward point for each transect.
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing time series data for each transect.
    - config: Configuration dictionary.

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides.
    """
    if seaward_points_gdf["region_id"].nunique() == 1:
        print("Modeling tides for all points")
        all_tides_df = model_tides_for_all(seaward_points_gdf, timeseries_df, config)
    else:
        print("Modeling tides by region ID")
        all_tides_df = model_tides_by_region_id(
            seaward_points_gdf, timeseries_df, config
        )
    return all_tides_df


def predict_tides(
    geojson_file_path: str,
    timeseries_df: pd.DataFrame,
    model_regions_geojson_path: str,
    config: dict,
) -> pd.DataFrame:
    """
    Predict tides based on input data and configurations.

    Parameters:
    - geojson_file_path: Path to the GeoJSON file containing transect data.
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing raw time series data.
    - model_regions_geojson_path: Path to the GeoJSON file containing model regions.
    - config: Configuration dictionary.

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides.
    """
    # get the seaward points from a geojson file
    
    
    # Load the GeoJSON file containing transect data
    seaward_points_gdf = read_and_filter_geojson(geojson_file_path)
    # Read in the model regions from a GeoJSON file
    regions_gdf = load_regions_from_geojson(model_regions_geojson_path)
    # Get the seaward points
    # seaward_points_gdf = get_seaward_points_gdf(transects_gdf)
    # Perform a spatial join to get the region_id for each point in seaward_points_gdf
    regional_seaward_points_gdf = perform_spatial_join(seaward_points_gdf, regions_gdf)
    # predict the tides
    all_tides_df = handle_tide_predictions(
        regional_seaward_points_gdf, timeseries_df, config
    )
    return all_tides_df


def convert_transect_ids_to_rows(df):
    """
    Reshapes the timeseries data so that transect IDs become rows.

    Args:
    - df (DataFrame): Input data with transect IDs as columns.

    Returns:
    - DataFrame: Reshaped data with transect IDs as rows.
    """
    reshaped_df = df.melt(
        id_vars="dates", var_name="transect_id", value_name="cross_distance"
    )
    return reshaped_df.dropna()


def apply_tide_correction(
    df: pd.DataFrame, reference_elevation: float, beach_slope: float
):
    """
    Applies tidal correction to the timeseries data.

    Args:
    - df (DataFrame): Input data with tide predictions and timeseries data.
    - reference_elevation (float): Reference elevation value.
    - beach_slope (float): Beach slope value.

    Returns:
    - DataFrame: Tidally corrected data.
    """
    correction = (df["tide"] - reference_elevation) / beach_slope
    df["cross_distance"] = df["cross_distance"] + correction
    return df.drop(columns=["correction"], errors="ignore")


def merge_dataframes(df1, df2, columns_to_merge_on=set(["transect_id", "dates"])):
    """
    Merges two DataFrames based on column names provided in columns_to_merge_on by default
    merges on "transect_id", "dates".

    Args:
    - df1 (DataFrame): First DataFrame.
    - df2 (DataFrame): Second DataFrame.
    - columns_to_merge_on(collection): column names to merge on
    Returns:
    - DataFrame: Merged data.
    """
    merged_df = pd.merge(df1, df2, on=list(columns_to_merge_on), how="inner")
    return merged_df.drop_duplicates(ignore_index=True)


def read_csv(file_path):
    """
    Reads a CSV file into a DataFrame and performs necessary preprocessing.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - DataFrame: Processed data.
    """
    df = pd.read_csv(file_path, parse_dates=["dates"])
    df["dates"] = pd.to_datetime(df["dates"], utc=True)
    for column in ["x", "y", "Unnamed: 0"]:
        if column in df.columns:
            df.drop(columns=column, inplace=True)
    return df


def get_location(filename: str, check_parent_directory: bool = False) -> Path:
    """
    Get the absolute path to a specified file.

    The function searches for the file in the directory where the script is located.
    Optionally, it can also check the parent directory.

    Parameters:
    - filename (str): The name of the file for which the path is sought.
    - check_parent_directory (bool): If True, checks the parent directory for the file.
      Default is False.

    Returns:
    - Path: The absolute path to the file.

    Raises:
    - FileNotFoundError: If the file is not found in the specified location(s).
    """
    try:
        search_dir = Path(__file__).parent
    except:
        # search_dir = Path(os.getcwd()).parent
        search_dir = Path(os.getcwd())

    if check_parent_directory:
        # Move up to the parent directory and then to 'tide_model'
        file_path = search_dir.parent / filename
    else:
        file_path = search_dir / filename
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    return file_path


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the tide correction script.

    Arguments and their defaults are defined within the function.

    Returns:
    - argparse.Namespace: A namespace containing the script's command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script to correct tides.")
    # DEFAULT LOCATIONS OF FES 2014 TIDE MODEL
    
    try:
        MODEL_REGIONS_GEOJSON_PATH = get_location("tide_regions_map.geojson")
    except FileNotFoundError as e:
        print("Model regions geojson file not found at the default location. Please provide the path manually.")
        MODEL_REGIONS_GEOJSON_PATH = ""
    try:
        FES_2014_MODEL_PATH = get_location("tide_model", check_parent_directory=True)
    except FileNotFoundError as e:
        print("FES 2014 tide model not found at the default location. Please provide the path manually.")
        FES_2014_MODEL_PATH = ""
        
        
    parser.add_argument(
        "-C",
        "-c",
        dest="config",
        type=str,
        required=True,
        help="Set the GEOJSON_FILE_PATH.",
    )
    parser.add_argument(
        "-T",
        "-t",
        dest="timeseries",
        type=str,
        required=True,
        help="Set the RAW_TIMESERIES_FILE_PATH.",
    )
    parser.add_argument(
        "-P",
        "-p",
        dest="predictions",
        type=str,
        default="tidal_predictions.csv",
        help="Set the TIDE_PREDICTIONS_FILE_NAME.",
    )
    parser.add_argument(
        "-R",
        "-r",
        dest="regions",
        type=str,
        default=MODEL_REGIONS_GEOJSON_PATH,
        help="Set the MODEL_REGIONS_GEOJSON_PATH.",
    )
    parser.add_argument(
        "-M",
        "-m",
        dest="model",
        type=str,
        default=FES_2014_MODEL_PATH,
        help="Set the FES_2014_MODEL_PATH.",
    )
    return parser.parse_args()


def setup_tide_model_config(model_path: str) -> dict:
    return {
        "DIRECTORY": model_path,
        "DELTA_TIME": [0],
        "GZIP": False,
        "MODEL": "FES2014",
        "ATLAS_FORMAT": "netcdf",
        "EXTRAPOLATE": True,
        "METHOD": "spline",
        "TYPE": "drift",
        "TIME": "datetime",
        "EPSG": 4326,
        "FILL_VALUE": np.nan,
        "CUTOFF": 10,
        "METHOD": "bilinear",
        "REGION_DIRECTORY": os.path.join(model_path, "region"),
    }


def main():
    args = parse_arguments()
    # Assigning the parsed arguments to the respective variables
    GEOJSON_FILE_PATH = args.config
    # Set this manually for testing only
    # GEOJSON_FILE_PATH = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\fire_island\ID_ham1_datetime08-03-23__10_58_34\config_gdf.geojson"
    RAW_TIMESERIES_FILE_PATH = args.timeseries
    # Set this manually for testing only
    # RAW_TIMESERIES_FILE_PATH = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\fire_island\ID_ham1_datetime08-03-23__10_58_34\raw_transect_time_series.csv"
    TIDE_PREDICTIONS_FILE_NAME = args.predictions
    MODEL_REGIONS_GEOJSON_PATH = args.regions
    FES_2014_MODEL_PATH = args.model
    
    if not os.path.exists(GEOJSON_FILE_PATH):
        raise FileNotFoundError(f"GeoJSON file not found: {GEOJSON_FILE_PATH}")
    
    if not os.path.exists(FES_2014_MODEL_PATH):
        raise FileNotFoundError(f"FES 2014 model directory not found: {FES_2014_MODEL_PATH}")
        
    if not os.path.exists(MODEL_REGIONS_GEOJSON_PATH):
        raise FileNotFoundError(f"Model regions directory not found: {MODEL_REGIONS_GEOJSON_PATH}")
    
    if not os.path.exists(RAW_TIMESERIES_FILE_PATH):
        raise FileNotFoundError(f"Time series data file not found: {RAW_TIMESERIES_FILE_PATH}")

    tide_model_config = setup_tide_model_config(FES_2014_MODEL_PATH)

    # Read timeseries data
    raw_timeseries_df = read_csv(RAW_TIMESERIES_FILE_PATH)
    print(raw_timeseries_df.head())
    # Optionlly read tide predictions from another session
    # predicted_tides_df = read_csv(TIDE_PREDICTIONS_FILE_NAME)

    print("Predicting tides this may take some time....")
    predicted_tides_df = predict_tides(
        GEOJSON_FILE_PATH,
        raw_timeseries_df,
        MODEL_REGIONS_GEOJSON_PATH,
        tide_model_config,
    )
    # drop the x and y columns (optional) (uncomment the line below to drop the x and y columns from the output file)
    # predicted_tides_df.drop(columns=['x','y'], inplace=True)
    
    predicted_tides_df.to_csv(TIDE_PREDICTIONS_FILE_NAME, index=False)
    print(f"Tide predictions saved to {os.path.abspath(TIDE_PREDICTIONS_FILE_NAME)}")

if __name__ == "__main__":
    main()
