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



def get_closest_slope(csv_file_path, transect_id, target_date:str, default_slope=0.0):
    provided_date = pd.to_datetime(target_date)
    df = pd.read_csv(csv_file_path, header=None)

    # Rename the first column to 'date' for better readability
    df.rename(columns={0: 'date'}, inplace=True)

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    # set the date column as the index
    df.set_index('date', inplace=True)
    # get the first row and turn it into the columns exception the first column
    df.columns = df.iloc[0].values
    # drop the first row because it contains the column names
    df = df[1:]

    if transect_id not in df.columns:
        raise ValueError(f"Transect ID {transect_id} not found in the CSV file")
    if np.all(df[transect_id].isna()):
        raise ValueError(f"All slope values for transect {transect_id} are NaN")

    # Filter non-NA values for the given transect ID
    transect_id_dates = df[transect_id].dropna()

    # Find the index of the date closest to the provided date
    closest_date_idx = np.abs(transect_id_dates.index - provided_date).argmin()

    # Get the value (slope) at the closest date
    closest_slope = transect_id_dates.iloc[closest_date_idx]

    return closest_slope


def get_closest_slope_matrix(matrix_csv_file_path, target_date:str):
    provided_date = pd.to_datetime(target_date)
    matrix_df = pd.read_csv(matrix_csv_file_path, index_col=0, parse_dates=True)

    # Find the closest date for each transect
    closest_slopes = {}
    for transect_id in matrix_df.columns:
        transect_id_dates = matrix_df[transect_id].dropna()
        closest_date_idx = np.abs(transect_id_dates.index - provided_date).argmin()
        closest_slope = transect_id_dates.iloc[closest_date_idx]
        closest_slopes[transect_id] = closest_slope

    return closest_slopes

def add_shore_points_to_timeseries(timeseries_data: pd.DataFrame,
                                 transects: gpd.GeoDataFrame,)->pd.DataFrame:
    """
    Edits the transect_timeseries_merged.csv or transect_timeseries_tidally_corrected.csv
    so that there are additional columns with lat (shore_y) and lon (shore_x).
    
    
    inputs:
    timeseries_data (pd.DataFrame): dataframe containing the data from transect_timeseries_merged.csv
    transects (gpd.GeoDataFrame): geodataframe containing the transects 
    
    returns:
    pd.DataFrame: the new timeseries_data with the lat and lon columns
    """
    
    ##Gonna do this in UTM to keep the math simple...problems when we get to longer distances (10s of km)
    org_crs = transects.crs
    utm_crs = transects.estimate_utm_crs()
    transects_utm = transects.to_crs(utm_crs)
    
    ##need some placeholders
    shore_x_vals = [None]*len(timeseries_data)
    shore_y_vals = [None]*len(timeseries_data)
    timeseries_data['shore_x'] = shore_x_vals
    timeseries_data['shore_y'] = shore_y_vals

    ##loop over all transects
    for i in range(len(transects_utm)):
        transect = transects_utm.iloc[i]
        transect_id = transect['id']
        first = transect.geometry.coords[0]
        last = transect.geometry.coords[1]
        
        idx = timeseries_data['transect_id'].str.contains(transect_id)
        ##in case there is a transect in the config_gdf that doesn't have any intersections
        ##skip that transect
        if np.any(idx):
            timeseries_data_filter = timeseries_data[idx]
        else:
            continue

        idxes = timeseries_data_filter.index
        distances = timeseries_data_filter['cross_distance']

        angle = np.arctan2(last[1] - first[1], last[0] - first[0])

        shore_x_utm = first[0]+distances*np.cos(angle)
        shore_y_utm = first[1]+distances*np.sin(angle)
        points_utm = [shapely.Point(xy) for xy in zip(shore_x_utm, shore_y_utm)]

        #conversion from utm to wgs84, put them in the transect_timeseries csv and utm gdf
        dummy_gdf_utm = gpd.GeoDataFrame({'geometry':points_utm},
                                         crs=utm_crs)
        dummy_gdf_wgs84 = dummy_gdf_utm.to_crs(org_crs)

        points_wgs84 = [shapely.get_coordinates(p) for p in dummy_gdf_wgs84.geometry]
        points_wgs84 = np.array(points_wgs84)
        points_wgs84 = points_wgs84.reshape(len(points_wgs84),2)
        x_wgs84 = points_wgs84[:,0]
        y_wgs84 = points_wgs84[:,1]
        timeseries_data.loc[idxes,'shore_x'] = x_wgs84
        timeseries_data.loc[idxes,'shore_y'] = y_wgs84

    return timeseries_data

def intersect_with_buffered_transects(points_gdf, transects, buffer_distance = 0.00000001):
    """
    Intersects points from a GeoDataFrame with another GeoDataFrame and exports the result to a new GeoDataFrame, retaining all original attributes.
    Additionally, returns the points that do not intersect with the buffered transects.
    
    Parameters:
    - points_gdf: GeoDataFrame - The input GeoDataFrame containing the points to be intersected.
    - transects: GeoDataFrame - The GeoDataFrame representing the transects to intersect with.
    - buffer_distance: float - The buffer distance to apply to the transects (default: 0.00000001).
    
    Returns:
    - filtered: GeoDataFrame - The resulting GeoDataFrame containing the intersected points within the buffered transects.
    - dropped_rows: GeoDataFrame - The rows that were filtered out during the intersection process.
    """
    
    buffered_lines_gdf = transects.copy()  # Create a copy to preserve the original data
    buffered_lines_gdf['geometry'] = transects.geometry.buffer(buffer_distance)
    points_within_buffer = points_gdf[points_gdf.geometry.within(buffered_lines_gdf.unary_union)]
    
    grouped = points_within_buffer.groupby('transect_id')
    
    # Filter out points not within their respective buffered transect
    filtered = grouped.filter(lambda x: x.geometry.within(buffered_lines_gdf[buffered_lines_gdf['id'].isin(x['transect_id'])].unary_union).all())

    # Identify the dropped rows by comparing the original dataframe within the buffer and the filtered results
    dropped_rows = points_gdf[~points_gdf.index.isin(filtered.index)]

    return filtered, dropped_rows

def filter_points_outside_transects(merged_timeseries_gdf:gpd.GeoDataFrame, transects_gdf:gpd.GeoDataFrame, save_location: str, name: str = ""):
    """
    Filters points outside of transects from a merged timeseries GeoDataFrame.

    Args:
        merged_timeseries_gdf (GeoDataFrame): The merged timeseries GeoDataFrame containing the shore x and shore y columns that indicated where the shoreline point was along the transect
        transects_gdf (GeoDataFrame): The transects GeoDataFrame used for filtering.
        save_location (str): The directory where the filtered points will be saved.
        name (str, optional): The name to be appended to the saved file. Defaults to "".

    Returns:
        tuple: A tuple containing the filtered merged timeseries GeoDataFrame and a DataFrame of dropped points.

    """
    extension = "" if name == "" else f'{name}_'
    timeseries_df = pd.DataFrame(merged_timeseries_gdf)
    timeseries_df.drop(columns=['geometry'], inplace=True)
    # estimate crs of transects
    utm_crs = merged_timeseries_gdf.estimate_utm_crs()
    # intersect the points with the transects
    filtered_merged_timeseries_gdf_utm, dropped_points_df = intersect_with_buffered_transects(
        merged_timeseries_gdf.to_crs(utm_crs), transects_gdf.to_crs(utm_crs))
    # Get a dataframe containing the points that were filtered out from the time series because they were not on the transects
    dropped_points_df.drop(columns=['geometry']).to_csv(os.path.join(save_location, f"{extension}dropped_points_time_series.csv"), index=False)
    # convert back to same crs as original merged_timeseries_gdf
    merged_timeseries_gdf = filtered_merged_timeseries_gdf_utm.to_crs(merged_timeseries_gdf.crs)
    return merged_timeseries_gdf, dropped_points_df


def filter_dropped_points_out_of_timeseries(timeseries_df:pd.DataFrame, dropped_points_df:pd.DataFrame)->pd.DataFrame:
    """
    Filter out dropped points from a timeseries dataframe.

    Args:
        timeseries_df (pandas.DataFrame): The timeseries dataframe to filter.
        dropped_points_df (pandas.DataFrame): The dataframe containing dropped points information.

    Returns:
        pandas.DataFrame: The filtered timeseries dataframe with dropped points set to NaN.
    """
    # Iterate through unique transect ids from drop_df to avoid setting the same column multiple times
    for t_id in dropped_points_df['transect_id'].unique():
        # Find all the dates associated with this transect_id in dropped_points_df
        dates_to_drop = dropped_points_df.loc[dropped_points_df['transect_id'] == t_id, 'dates']
        timeseries_df.loc[timeseries_df['dates'].isin(dates_to_drop), t_id] = np.nan
    return timeseries_df

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
    linestrings.to_crs(output_crs, inplace=True)
    linestrings_gdf = gpd.GeoDataFrame(linestrings, columns=['geometry'], crs=output_crs)
    linestrings_gdf.reset_index(inplace=True)
    
    # order the linestrings so that they are continuous
    linestrings_gdf = order_linestrings_gdf(linestrings_gdf,linestrings_gdf['date'],output_crs=output_crs)
    
    return linestrings_gdf

def add_lat_lon_to_timeseries(merged_timeseries_df, transects_gdf,timeseries_df,
                              save_location:str,
                              only_keep_points_on_transects:bool=False,
                              extension:str=""):
    """
    Adds latitude and longitude coordinates to a timeseries dataframe based on shoreline positions.

    Args:
        merged_timeseries_df (pandas.DataFrame): The timeseries dataframe to add latitude and longitude coordinates to.
        transects_gdf (geopandas.GeoDataFrame): The geodataframe containing transect information.
        timeseries_df (pandas.DataFrame): The original timeseries dataframe.This is a matrix of dates x transect id with the cross shore distance as the values.
        save_location (str): The directory path to save the output files.
        only_keep_points_on_transects (bool, optional): Whether to keep only the points that fall on the transects. 
                                                  Defaults to False.
        extension (str, optional): An extension to add to the output filenames. Defaults to "".
        

    Returns:
        pandas.DataFrame: The updated timeseries dataframe with latitude and longitude coordinates.

    """
    ext = "" if extension=="" else f'{extension}'
    
    # add the shoreline position as an x and y coordinate to the csv called shore_x and shore_y
    merged_timeseries_df = add_shore_points_to_timeseries(merged_timeseries_df, transects_gdf)
    # convert to geodataframe
    merged_timeseries_gdf = gpd.GeoDataFrame(
        merged_timeseries_df, 
        geometry=[Point(xy) for xy in zip(merged_timeseries_df['shore_x'], merged_timeseries_df['shore_y'])], 
        crs="EPSG:4326"
    )
    if only_keep_points_on_transects:
        merged_timeseries_gdf,dropped_points_df = filter_points_outside_transects(merged_timeseries_gdf,transects_gdf,save_location,ext)
        if not dropped_points_df.empty:
            timeseries_df = filter_dropped_points_out_of_timeseries(timeseries_df, dropped_points_df)
    
    # save the time series of along shore points as points to a geojson (saves shore_x and shore_y as x and y coordinates in the geojson)
    cross_shore_pts = convert_date_gdf(merged_timeseries_gdf.drop(columns=['x','y','shore_x','shore_y','cross_distance']).to_crs('epsg:4326'))
    # rename the dates column to date
    cross_shore_pts.rename(columns={'dates':'date'},inplace=True)
    
    # Create 2D vector of shorelines from where each shoreline intersected the transect
    if cross_shore_pts.empty:
        print("No points were found on the transects. Skipping the creation of the transect_time_series_points.geojson and transect_time_series_vectors.geojson files")
        return merged_timeseries_df,timeseries_df
    
    # convert the cross shore points to crs 4326 before converting to linestrings
    cross_shore_pts.to_crs('epsg:4326', inplace=True)
    new_gdf_shorelines_wgs84=convert_points_to_linestrings(cross_shore_pts, group_col='date', output_crs='epsg:4326')
    new_gdf_shorelines_wgs84_path = os.path.join(save_location, f'{ext}_transect_time_series_vectors.geojson')
    new_gdf_shorelines_wgs84.to_file(new_gdf_shorelines_wgs84_path)
    
    # save the merged time series that includes the shore_x and shore_y columns to a geojson file and a  csv file
    merged_timeseries_gdf_cleaned = convert_date_gdf(merged_timeseries_gdf.drop(columns=['x','y','shore_x','shore_y','cross_distance']).rename(columns={'dates':'date'}).to_crs('epsg:4326'))
    merged_timeseries_gdf_cleaned.to_file(os.path.join(save_location, f"{ext}_transect_time_series_points.geojson"), driver='GeoJSON')
    merged_timeseries_df = pd.DataFrame(merged_timeseries_gdf.drop(columns=['geometry']))

    return merged_timeseries_df,timeseries_df

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
    # if the transect ID is not in the timeseries_df then return None
    if transect_id != "":
        if transect_id not in timeseries_df.columns:
            return None
        dates_for_transect_id_df = timeseries_df[["dates", transect_id]].dropna()
    tide_predictions_df = model_tides(
        x,
        y,
        dates_for_transect_id_df.dates.values,
        transect_id=transect_id,
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
                                         f"{region_directory}{row['region_id']}",
                                         row["transect_id"]), axis=1
    )
    # Filter out None values
    all_tides = all_tides.dropna()
    # if no tides are predicted return an empty dataframe
    if all_tides.empty:
        return pd.DataFrame(columns=["dates", "x", "y", "tide", "transect_id"])
    
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
    feature_type: str = "transect",
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
    # Filter the features with "type" equal to the specified feature_type
    filtered_gdf = gdf[gdf["type"] == feature_type]

    return filtered_gdf


def load_regions_from_geojson(geojson_path: str) -> gpd.GeoDataFrame:
    """
    Load regions from a GeoJSON file and assign a region_id based on index.

    Parameters:
    - geojson_path: Path to the GeoJSON file containing regions.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the loaded regions with an added 'region_id' column.
    """
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
        all_tides_df = model_tides_for_all(seaward_points_gdf, timeseries_df, config)
    else:
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
    # Load the GeoJSON file containing transect data
    transects_gdf = read_and_filter_geojson(geojson_file_path)
    # Read in the model regions from a GeoJSON file
    regions_gdf = load_regions_from_geojson(model_regions_geojson_path)
    # Get the seaward points
    seaward_points_gdf = get_seaward_points_gdf(transects_gdf)
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


def tidally_correct_timeseries(
    timeseries_df: pd.DataFrame,
    tide_predictions_df: pd.DataFrame,
    reference_elevation: float,
    beach_slope: float,
) -> pd.DataFrame:
    """
    Applies tidal correction to the timeseries data and saves the tidally corrected data to a csv file

    Args:
    - timeseries_df (pd.DataFrame): A DataFrame containing time series data for each transect. A DataFrame containing raw time series data.
    - tide_predictions_df (pd.DataFrame):A DataFrame containing predicted tides each transect's seaward point in the timeseries
    - reference_elevation (float): Reference elevation value.
    - beach_slope (float): Beach slope value.

    Returns:
        - pd.DataFrame: A DataFrame containing the timeseries_df that's been tidally corrected with the predicted tides
    """
    timeseries_df = convert_transect_ids_to_rows(timeseries_df)
    merged_df = merge_dataframes(tide_predictions_df, timeseries_df)
    corrected_df = apply_tide_correction(merged_df, reference_elevation, beach_slope)
    return corrected_df


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
    search_dir = Path(__file__).parent
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

    MODEL_REGIONS_GEOJSON_PATH = get_location("tide_regions_map.geojson")
    FES_2014_MODEL_PATH = get_location("tide_model", check_parent_directory=True)

    parser.add_argument(
        "-C",
        "-c",
        dest="config",
        type=str,
        required=True,
        help="Set the CONFIG_FILE_PATH.",
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
        "-E",
        "-e",
        dest="elevation",
        type=float,
        default=3,
        help="Set the REFERENCE_ELEVATION.",
    )
    parser.add_argument(
        "-S", "-s", dest="slope", type=float, default=2, help="Set the BEACH_SLOPE."
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
        "-O",
        "-o",
        dest="output",
        type=str,
        default="tidally_corrected_transect_time_series.csv",
        help="Set the TIDALLY_CORRECTED_FILE_NAME.",
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
    parser.add_argument(
        "-D",
        "-d",
        dest="drop",
        action="store_true",
        default=False,
        help="Add -d to drop any shoreline transect intersections points that are not on the transect from the csv files",
    )
    parser.add_argument(
        "--matrix",
        "-m",
        dest="matrix",
        type=str,
        help="Path to the matrix CSV file with transect ID as columns and dates as rows",
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
    REFERENCE_ELEVATION = args.elevation
    BEACH_SLOPE = args.slope
    CONFIG_FILE_PATH = args.config
    # Set this manually for testing only
    # CONFIG_FILE_PATH = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\fire_island\ID_ham1_datetime08-03-23__10_58_34\config_gdf.geojson"
    RAW_TIMESERIES_FILE_PATH = args.timeseries
    # Set this manually for testing only
    # RAW_TIMESERIES_FILE_PATH = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\fire_island\ID_ham1_datetime08-03-23__10_58_34\raw_transect_time_series.csv"
    TIDE_PREDICTIONS_FILE_NAME = args.predictions
    TIDALLY_CORRECTED_FILE_NAME = args.output
    TIDALLY_CORRECTED_MATRIX_FILE_NAME = 'tidally_corrected_transect_time_series.csv'
    MODEL_REGIONS_GEOJSON_PATH = args.regions
    FES_2014_MODEL_PATH = args.model
    DROP_INTERSECTIONS = args.drop
    MATRIX_CSV_FILE_PATH = args.matrix

    tide_model_config = setup_tide_model_config(FES_2014_MODEL_PATH)

    if MATRIX_CSV_FILE_PATH and os.path.isfile(MATRIX_CSV_FILE_PATH):
        target_date = '2011-08-02'  # Example date, you can change as needed
        closest_slopes = get_closest_slope_matrix(MATRIX_CSV_FILE_PATH, target_date)
        print("Closest slopes for each transect:")
        for transect, slope in closest_slopes.items():
            print(f"Transect ID {transect}: Closest Slope {slope}")
        return

    # Read timeseries data
    raw_timeseries_df = read_csv(RAW_TIMESERIES_FILE_PATH)
    # Optionally read tide predictions from another session
    # predicted_tides_df = read_csv(TIDE_PREDICTIONS_FILE_NAME)

    start_time = time.time()
    print("Predicting tides this may take some time....")
    predicted_tides_df = predict_tides(
        CONFIG_FILE_PATH,
        raw_timeseries_df,
        MODEL_REGIONS_GEOJSON_PATH,
        tide_model_config,
    )
    end_time = time.time()
    print(f"Time taken for all tide predictions: {end_time - start_time}s")
    # Save the tide_predictions
    print(f"Predicted tides saved to {os.path.abspath(TIDE_PREDICTIONS_FILE_NAME)}")

    # format the predicted tides as a matrix of date vs transect id with the tide as the values
    # Pivot the table
    pivot_df = predicted_tides_df.pivot_table(index='dates', columns='transect_id', values='tide', aggfunc='first')
    # Reset index if you want 'dates' back as a column
    pivot_df.reset_index(inplace=True)

    pivot_df.to_csv(TIDE_PREDICTIONS_FILE_NAME, index=False)

    print(f"Applying tide corrections to {RAW_TIMESERIES_FILE_PATH}")
    tide_corrected_timeseries_df = tidally_correct_timeseries(
        raw_timeseries_df,
        predicted_tides_df,
        REFERENCE_ELEVATION,
        BEACH_SLOPE,
    )

    # Load the GeoJSON file containing transect data
    transects_gdf = read_and_filter_geojson(CONFIG_FILE_PATH)

    tide_corrected_timeseries_df_matrix = tide_corrected_timeseries_df.pivot_table(index='dates', columns='transect_id', values='cross_distance', aggfunc='first')
    # Reset index if you want 'dates' back as a column
    tide_corrected_timeseries_df_matrix.reset_index(inplace=True)

    # Add lat lon to the timeseries data
    tide_corrected_timeseries_df, tide_corrected_timeseries_df_matrix = add_lat_lon_to_timeseries(tide_corrected_timeseries_df, transects_gdf, tide_corrected_timeseries_df_matrix, os.getcwd(),
                                                                                                 only_keep_points_on_transects=DROP_INTERSECTIONS, extension='tidally_corrected')
    # optionally save to session location in ROI the tide_corrected_timeseries_df to csv
    tide_corrected_timeseries_df.to_csv(
        os.path.join(os.getcwd(), "tidally_corrected_transect_time_series_merged.csv"), index=False
    )
    # Tidally correct the raw time series
    print(f"Tidally corrected data saved to {os.path.abspath(TIDALLY_CORRECTED_FILE_NAME)}")
    # Save the Tidally corrected time series
    tide_corrected_timeseries_df.to_csv(TIDALLY_CORRECTED_FILE_NAME, index=False)

    # save the time series as a matrix of date vs transect id with the cross_distance as the values
    pivot_df = tide_corrected_timeseries_df.pivot_table(index='dates', columns='transect_id', values='cross_distance', aggfunc='first')

    # Reset index if you want 'dates' back as a column
    pivot_df.reset_index(inplace=True)
    # Tidally correct the raw time series
    print(f"Tidally corrected data saved to {os.path.abspath(TIDALLY_CORRECTED_MATRIX_FILE_NAME)}")
    # Save the Tidally corrected time series
    pivot_df.to_csv(TIDALLY_CORRECTED_MATRIX_FILE_NAME, index=False)



if __name__ == "__main__":
    main()
