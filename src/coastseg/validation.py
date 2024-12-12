import os
from enum import Enum
import re

class Satellite(Enum):
    L5 = 'L5'
    L7 = 'L7'
    L8 = 'L8'
    L9 = 'L9'
    S2 = 'S2'
    
    
def is_valid_satellite(satellite_name: str) -> bool:
    return satellite_name.upper() in (sat.value.upper() for sat in Satellite)



def find_satellite_in_filename(filename: str) -> str:
    """Use regex to find the satellite name in the filename.
    Satellite name is case-insensitive and can be separated by underscore (_) or period (.)"""
    for satellite in Satellite:
        # Adjusting the regex pattern to consider period (.) as a valid position after the satellite name
        if re.search(fr'(?<=[\b_]){satellite.value}(?=[\b_.]|$)', filename, re.IGNORECASE):
            return satellite.value
    return ""


def get_satellites_in_directory(directory_path: str) -> set:
    """Get the set of satellite names for each folder in the directory path."""
    satellite_set = set()

    for filename in os.listdir(directory_path):
        satellite_name = find_satellite_in_filename(filename)
        if satellite_name:
            satellite_set.add(satellite_name)

    return satellite_set


