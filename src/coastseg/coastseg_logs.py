import os
from datetime import datetime
import logging
from coastseg import core_utilities
# DESCRIPTION: Sets up a logging system that writes logs to a file named with the current timestamp in a "logs" directory.


def prepare_logging():
    """create a directory named 'logs' in the current working directory if a 'logs' directory does not exist"""
    if not os.path.exists(os.path.abspath(os.path.join(core_utilities.get_base_dir(), "logs"))):
        os.mkdir(os.path.abspath(os.path.join(core_utilities.get_base_dir(), "logs")))


def clear_default_handlers() -> None:
    """Removes the default logging handlers"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def create_root_logger():
    """Creates the root logger. The root logger will write to a log file with the format
    log_<year>-<month>-<day>-<hour>-<minute>. This log file will  be written to by all the other loggers
    """
    log_filename = "log_" + datetime.now().strftime("%m-%d-%y-%I_%M_%S") + ".txt"
    log_file = os.path.abspath(os.path.join(core_utilities.get_base_dir(), "logs", log_filename))
    # configure the logger
    log_format = "%(asctime)s - %(filename)s at line %(lineno)s in %(funcName)s() - %(levelname)s : %(message)s"
    os.path.abspath(os.path.join(core_utilities.get_base_dir(), "logs"))
    # Use FileHandler() to log to a file
    file_handler = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    # Have all loggers write to the same log file
    logging.basicConfig(
        handlers=[file_handler],
        format=log_format,
        level=logging.INFO,
        datefmt="-%m-%d-%y-%I:%M:%S",
    )


# Prepare and create the logger
prepare_logging()
# clear the default logging handlers this is needed for the logs to work in google colab
clear_default_handlers()
create_root_logger()
