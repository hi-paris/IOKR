# Store all the support functions

from pathlib import Path
import os


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def create_path_that_doesnt_exist(path_save: str, file_name: str, extension: str):
    if not os.path.isdir(path_save):
        os.makedirs(path_save)
    # Increment a counter so that previous results with the same args will not
    # be overwritten. Comment out the next four lines if you only want to keep
    # the most recent results.
    i = 0
    while os.path.exists(os.path.join(path_save, file_name + str(i) + extension)):
        i += 1
    return os.path.join(path_save, file_name + str(i) + extension)
