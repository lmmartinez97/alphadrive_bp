import os
import datetime

def create_directory(path):
    """
    Checks if a directory exists. If it does not, it creates it.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_logging_directory(path):
    """
    Checks if a logging directory with the current timestamp exists. If it does not, it creates it.
    """
    path = os.path.join(path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(path):
        os.makedirs(path)
    return path