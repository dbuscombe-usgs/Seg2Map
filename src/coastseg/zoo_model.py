import os
import glob
import asyncio
import platform
import json
import logging
import requests
import aiohttp
import tqdm
import tqdm.asyncio
from tensorflow.python.client import device_lib
from tensorflow.keras import mixed_precision
from doodleverse_utils.prediction_imports import do_seg
from doodleverse_utils.imports import (
    simple_resunet,
    simple_unet,
    custom_resunet,
    custom_unet,
)
from doodleverse_utils.model_imports import dice_coef_loss, iou_multi, dice_multi
import tensorflow as tf

logger = logging.getLogger(__name__)

async def fetch(session,url:str,save_path: str):
    chunk_size: int = 128
    async with session.get(url,raise_for_status=True) as r:
        with open(save_path, "wb") as fd:
            async for chunk in r.content.iter_chunked(chunk_size):
                fd.write(chunk)

async def fetch_all(session,url_dict):
    tasks = []
    for save_path,url in url_dict.items():
        task = asyncio.create_task(fetch(session,url,save_path)) 
        tasks.append(task)
    await tqdm.asyncio.tqdm.gather(*tasks)

async def async_download_urls(url_dict:dict)->None:
    async with aiohttp.ClientSession() as session:
        await fetch_all(session,url_dict)

def run_async_download(url_dict:dict):
    logger.info("run_async_download")
    if platform.system()=='Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop()
    loop.create_task(async_download_urls(url_dict))


def get_GPU(use_GPU:bool)->None:
    if use_GPU == False:
        logger.info("Not using GPU")
        print("Not using GPU")
        # use CPU (not recommended):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print(f"physical_devices (GPUs):{physical_devices}")
        logger.info(f"physical_devices (GPUs):{physical_devices}")
    elif use_GPU == True:
        print("Using  GPU")
        logger.info("Using  GPU")
        # use first available GPU (@todo I think this line was set just for testing change back to 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        # read physical GPUs from machine
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print(f"physical_devices (GPUs):{physical_devices}")
        logger.info(f"physical_devices (GPUs):{physical_devices}")
        if physical_devices:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
            except RuntimeError as e:
                # Visible devices must be set at program startup
                logger.error(e)
                print(e)
    # set mixed precision
    mixed_precision.set_global_policy('mixed_float16')
    # disable memory growth on all GPUs
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)
        print(f"visible_devices: {tf.config.get_visible_devices()}")
        logger.info(f"visible_devices: {tf.config.get_visible_devices()}")

def get_url_dict_to_download(models_json_dict:dict)->dict:
    """Returns dictionary of paths to save files to download
    and urls to download file
    
    ex.
    {'C:\Home\Project\file.json':"https://website/file.json"}

    Args:
        models_json_dict (dict): full path to files and links

    Returns:
        dict: full path to files and links
    """    
    url_dict={}
    print(models_json_dict)
    for save_path,link in models_json_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path]=link
        json_filepath = save_path.replace("_fullmodel.h5",".json")
        if not os.path.isfile(json_filepath):
            json_link = link.replace("_fullmodel.h5",".json")
            url_dict[json_filepath]=json_link
            
    return url_dict
    
def download_url(url: str, save_path: str, chunk_size: int = 128):
    """Downloads the model from the given url to the save_path location.
    Args:
        url (str): url to model to download
        save_path (str): directory to save model
        chunk_size (int, optional):  Defaults to 128.
    """
    logger.info(f"url: {url}")
    logger.info(f"save_path: {save_path}")
    # make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # check header to get content length, in bytes
        content_length = r.headers.get("Content-Length")
        # raise an exception for error codes (4xx or 5xx)
        r.raise_for_status()
        if content_length is None:
            with open(save_path, "wb") as fd:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
        elif content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with tqdm.auto.tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading Model",
                    initial=0,
                    ascii=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))


class Zoo_Model:
    def __init__(self):
        self.weights_direc = None

    def get_files_for_seg(self, sample_direc: str) -> list:
        """Returns list of files to be segmented
        Args:
            sample_direc (str): directory containing files to be segmented

        Returns:
            list: files to be segmented
        """
        # Read in the image filenames as either .npz,.jpg, or .png
        sample_filenames = sorted(glob.glob(sample_direc + os.sep + "*.*"))
        if sample_filenames[0].split(".")[-1] == "npz":
            sample_filenames = sorted(tf.io.gfile.glob(sample_direc + os.sep + "*.npz"))
        else:
            sample_filenames = sorted(tf.io.gfile.glob(sample_direc + os.sep + "*.jpg"))
            if len(sample_filenames) == 0:
                sample_filenames = sorted(glob.glob(sample_direc + os.sep + "*.png"))
        return sample_filenames

    def compute_segmentation(
        self,
        sample_direc: str,
        model_list: list,
        metadatadict: dict,
    ):
        # look for TTA config
        if "TESTTIMEAUG" not in locals():
            TESTTIMEAUG = False
        WRITE_MODELMETADATA = False
        # Read in the image filenames as either .npz,.jpg, or .png
        files_to_segment = self.get_files_for_seg(sample_direc)
        # Compute the segmentation for each of the files
        for file_to_seg in tqdm.auto.tqdm(files_to_segment):
            do_seg(
                file_to_seg,
                model_list,
                metadatadict,
                sample_direc=sample_direc,
                NCLASSES=self.NCLASSES,
                N_DATA_BANDS=self.N_DATA_BANDS,
                TARGET_SIZE=self.TARGET_SIZE,
                TESTTIMEAUG=TESTTIMEAUG,
                WRITE_MODELMETADATA=WRITE_MODELMETADATA,
                OTSU_THRESHOLD=False,
            )


    def get_model(self, weights_list: list):
        model_list = []
        config_files = []
        model_types = []
        if weights_list == []:
            raise Exception("No Model Info Passed")
        for weights in weights_list:
            # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
            # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
            configfile = weights.replace(".h5", ".json").replace("weights", "config")
            if "fullmodel" in configfile:
                configfile = configfile.replace("_fullmodel", "")
            with open(configfile) as f:
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
                # 242,812
                else:
                    raise Exception(
                        f"An unknown model type {MODEL} was received. Please select a valid model."
                    )

                # Load in the custom loss function from doodleverse_utils
                model.compile(
                    optimizer="adam", loss=dice_coef_loss(self.NCLASSES)
                )  # , metrics = [iou_multi(self.NCLASSESNCLASSES), dice_multi(self.NCLASSESNCLASSES)])

                model.load_weights(weights)

            model_types.append(MODEL)
            model_list.append(model)
            config_files.append(configfile)

        return model, model_list, config_files, model_types

    def get_metadatadict(self, weights_list: list, config_files: list, model_types: list):
        metadatadict = {}
        metadatadict["model_weights"] = weights_list
        metadatadict["config_files"] = config_files
        metadatadict["model_types"] = model_types
        return metadatadict

    def get_weights_list(self, model_choice: str = "ENSEMBLE"):
        """Returns of the weights files(.h5) within weights_direc"""
        if model_choice == "ENSEMBLE":
            weights_list = glob.glob(self.weights_direc + os.sep + "*.h5")
            logger.info(f"ENSEMBLE: weights_list: {weights_list}")
            logger.info(f"ENSEMBLE: {len(weights_list)} sets of model weights were found ")
            return weights_list
        elif model_choice == "BEST":
            # read model name (fullmodel.h5) from BEST_MODEL.txt
            with open(self.weights_direc + os.sep + "BEST_MODEL.txt") as f:
                model_name = f.readlines()
            weights_list = [self.weights_direc + os.sep + model_name[0]]
            logger.info(f"BEST: weights_list: {weights_list}")
            logger.info(f"BEST: {len(weights_list)} sets of model weights were found ")
            return weights_list

    def get_downloaded_models_dir(self) -> str:
        """returns full path to downloaded_models directory and
        if downloaded_models directory does not exist then it is created

        Returns:
            str: full path to downloaded_models directory
        """
        # directory to hold downloaded models from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        downloaded_models_path = os.path.abspath(
            os.path.join(script_dir, "downloaded_models")
        )
        if not os.path.exists(downloaded_models_path):
            os.mkdir(downloaded_models_path)
        logger.info(f"downloaded_models_path: {downloaded_models_path}")
        return downloaded_models_path

    def download_model(self, model_choice: str, dataset_id: str) -> None:
        """downloads model specified by zenodo id in dataset_id.

        Downloads best model is model_choice = 'BEST' or all models in
        zenodo release if model_choice = 'ENSEMBLE'

        Args:
            model_choice (str): 'BEST' or 'ENSEMBLE'
            dataset_id (str): name of model followed by underscore zenodo_id'name_of_model_zenodoid'
        """
        zenodo_id = dataset_id.split("_")[-1]
        root_url = "https://zenodo.org/api/records/" + zenodo_id
        # read raw json and get list of available files in zenodo release
        response = requests.get(root_url)
        json_content = json.loads(response.text)
        files = json_content["files"]

        downloaded_models_path = self.get_downloaded_models_dir()
        # directory to hold specific model referenced by dataset_id
        self.weights_direc = os.path.abspath(
            os.path.join(downloaded_models_path, dataset_id)
        )
        if not os.path.exists(self.weights_direc):
            os.mkdir(self.weights_direc)

        logger.info(f"self.weights_direc:{self.weights_direc}")
        print(f"\n Model located at: {self.weights_direc}")
        models_json_dict={}
        if model_choice.upper() == "BEST":
            # retrieve best model text file
            best_model_json = [f for f in files if f["key"] == "BEST_MODEL.txt"][0]
            logger.info(f"list of best_model_txt: {best_model_json}")
            best_model_txt_path = self.weights_direc + os.sep + "BEST_MODEL.txt"
            logger.info(f"BEST: best_model_txt_path : {best_model_txt_path }")
            
            # if best BEST_MODEL.txt file not exist then download it
            if not os.path.isfile(best_model_txt_path):
                download_url(
                    best_model_json["links"]["self"],
                    best_model_txt_path,
                )
            # read contents of BEST_MODEL.txt
            with open(best_model_txt_path) as f:
                filename = f.read()

            # check if json and h5 file in BEST_MODEL.txt exist
            model_json = [f for f in files if f["key"] == filename][0]
            # path to save model
            outfile = self.weights_direc + os.sep + filename
            logger.info(f"BEST: outfile: {outfile}")
            # path to save file and json data associated with file saved to dict
            models_json_dict[outfile]=model_json["links"]["self"]
            url_dict = get_url_dict_to_download(models_json_dict)
            print(url_dict)
            run_async_download(url_dict)
        elif model_choice.upper() == "ENSEMBLE":
            # get list of all models
            all_models = [f for f in files if f["key"].endswith(".h5")]
            logger.info(f"all_models : {all_models }")
            # check if all h5 files in files are in self.weights_direc
            for model_json in all_models:
                outfile = (
                    self.weights_direc + os.sep + model_json["links"]["self"].split("/")[-1]
                )
                logger.info(f"ENSEMBLE: outfile: {outfile}")
                # path to save file and json data associated with file saved to dict
                models_json_dict[outfile]=model_json["links"]["self"]
            print(models_json_dict)
            url_dict = get_url_dict_to_download(models_json_dict)
            print(url_dict)
            run_async_download(url_dict)
