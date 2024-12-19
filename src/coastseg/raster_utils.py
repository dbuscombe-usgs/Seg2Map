from osgeo import gdal
import numpy as np
# import rasterio

def read_bands(filename: str, satname: str = "") -> list:
    """
    Read all the raster bands of a geospatial image file using GDAL.

    This function opens the provided geospatial file using GDAL in read-only mode
    and extracts each of the raster bands as a separate array. Each array represents
    the values of the raster band across the entire spatial extent of the image.

    Parameters:
    -----------
    filename : str
        The path to the geospatial image file (e.g., a GeoTIFF) to be read.

    Returns:
    --------
    list of np.ndarray
        A list containing 2D numpy arrays. Each array in the list represents
        one band from the geospatial image. The order of the arrays in the list
        corresponds to the order of the bands in the original image.

    Notes:
    ------
    This function relies on GDAL for reading geospatial data. Ensure that GDAL
    is properly installed and available in the Python environment.

    Example:
    --------
    bands = read_bands('path/to/image.tif')
    """
    data = gdal.Open(filename, gdal.GA_ReadOnly)
    # save the contents of each raster band as an array and save each array to the bands list
    # save separate NumPy array for each raster band in the dataset, with each array representing the pixel values of the corresponding band
    if satname == "S2" and data.RasterCount == 5:
        bands = [
            data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount - 1)
        ]
    else:
        bands = [
            data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)
        ]
    return bands

def read_ms_planet(fn_ms):
    # read ms bands
    data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    bands = read_bands(fn_ms)
    im_ms = np.stack(bands, 2)
    return im_ms, georef

def read_cloud_mask_planet(filepath: str,cloud_mask_band = 6) -> np.ndarray:
    """The UDM mask in planet has a lot more data so the cloud mask is different"""
    data = gdal.Open(filepath, gdal.GA_ReadOnly)
    cloud_mask = data.GetRasterBand(cloud_mask_band).ReadAsArray()
    return cloud_mask

def read_nodata_mask_planet(filepath: str,nodata_band=8) -> np.ndarray:
    """The UDM mask in planet has a lot more data so the cloud mask is different"""
    data = gdal.Open(filepath, gdal.GA_ReadOnly)
    im_nodata = data.GetRasterBand(nodata_band).ReadAsArray()
    return im_nodata

# Example usage:
# udm_path = r"C:\development\coastseg-planet\downloads\UNALAKLEET_pier_cloud_0.7_TOAR_enabled_2020-06-01_to_2023-08-01\e2821741-0677-435a-a93a-d4f060f3adf4\coregistered\PS\udm2\2020-06-11-21-52-01_3B_AnalyticMS_toar_clip.tif"
# ms_path = r"C:\development\coastseg-planet\downloads\UNALAKLEET_pier_cloud_0.7_TOAR_enabled_2020-06-01_to_2023-08-01\e2821741-0677-435a-a93a-d4f060f3adf4\coregistered\PS\ms\2020-06-11-21-52-01_3B_AnalyticMS_toar_clip.tif"

# print(np.all(read_cloud_mask_planet(udm_path))) # it works!
# print(read_ms_planet(ms_path)) # it works!

