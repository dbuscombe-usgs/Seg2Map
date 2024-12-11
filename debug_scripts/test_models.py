import os
import argparse

# internal python imports
from coastseg import zoo_model
from transformers import TFSegformerForSemanticSegmentation

# Last update 11/04/2024
# Added new RGB segformer models to the available_models_dict
# removed old sat models from the available_models_dict


# how to run this script
# Replace parameter after -P with path to ROI's RGB directory
# python test_models.py -P <your path here>"
# Example
# python test_models.py -P C:\development\doodleverse\coastseg\CoastSeg\data\ID_12_datetime06-05-23__04_16_45\jpg_files\preprocessed\RGB"

# Create the parser
parser = argparse.ArgumentParser(description="Run models on provided input directory.")
parser.add_argument(
    "-P",
    "--path",
    type=str,
    help="Path to an ROI's RGB directory from the data directory",
)
args = parser.parse_args()

# get input directory from the args
input_directory = args.path

# uncomment the following lines for local testing only
CoastSeg_location = r"C:\development\doodleverse\coastseg\CoastSeg" # path to CoastSeg
roi_name = "ID_ztg2_datetime07-01-24__02_39_39" # name of the ROI directory in CoastSeg
input_directory = os.path.join(CoastSeg_location, "data", roi_name, "jpg_files", "preprocessed", "RGB") # this is the path to the RGB directory of the ROI

def test_model(img_types:list, available_models_dict:dict, input_directory:str):
    """
    Tests models on different types of images.
    Parameters:
    img_types (list): A list of image types to be tested.
    available_models_dict (dict): A dictionary where keys are image types and values are lists of models available for those image types.
    input_directory (str): The directory where input images are stored. Should be the RGB directory for the ROI containing the images 
    Raises:
    Exception: If session_name is not generated correctly.
    Prints:
    Various debug information including the session name, implementation type, selected image type, selected model, and sample directory.
    """
    for img_type_index in range(len(img_types)):
        print(f"Running models for image input type {img_type_index}")
        selected_img_type = img_types[img_type_index]
        model_list = available_models_dict.get(selected_img_type)
        if not model_list:
            print(f"No models available for {selected_img_type}")
            continue

        for model_selected in model_list:
            print(f"Running model {model_selected}")
            implementation = "BEST"  # "ENSEMBLE" or "BEST"
            session_name = (
                model_selected
                + "_"
                + selected_img_type
                + "_"
                + implementation
                + "_"
                + "session"
            )
            if not session_name:
                raise Exception("Something went wrong...")

            print(f"session_name: {session_name}")
            print(f"implementation: {implementation}")
            print(f"selected_img_type: {selected_img_type}")
            print(f"model_selected: {model_selected}")
            print(f"sample_directory: {input_directory}")

            # load the basic zoo_model settings
            model_dict = {
                "sample_direc": input_directory,
                "use_GPU": "0",
                "implementation": implementation,
                "model_type": model_selected,
                "otsu": False,
                "tta": False,
                "use_local_model": False,
            }

            zoo_model_instance = zoo_model.Zoo_Model()

            zoo_model_instance.run_model(
                img_types[img_type_index],
                model_dict["implementation"],
                session_name,
                model_dict["sample_direc"],
                model_name=model_dict["model_type"],
                use_GPU="0",
                use_otsu=model_dict["otsu"],
                use_tta=model_dict["tta"],
                percent_no_data=50.0,
            )


available_models_dict = {
    "RGB": [
        "global_segformer_RGB_4class_14036903", # global segformer model
        "AK_segformer_RGB_4class_14037041", # AK segformer model
    ],
    "MNDWI": [
        "global_segformer_MNDWI_4class_14183366", # global segformer model
        "AK_segformer_MNDWI_4class_14187478", # AK segformer model
    ],
    "NDWI": [
      "global_segformer_NDWI_4class_14172182", # global segformer model
      "AK_segformer_NDWI_4class_14183210", # AK segformer model
    ],
}



img_types = ["RGB", "MNDWI", "NDWI"]
# img_types = [ "NDWI"]
# img_types = [ "MNDWI"]

test_model(img_types, available_models_dict, input_directory)