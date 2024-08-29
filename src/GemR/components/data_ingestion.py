
import zipfile
import os
from GemR.logger import logging
import urllib.request as request
import requests
from GemR.utils.common import get_size
from GemR.entity.config_entity import (DataIngestionConfig)
from pathlib import Path
# from GemR.components.data_transformation import DataTransformation, DataTransformationConfig
# from GemR.components.model_trainer import ModelTrainerConfig, ModelTrainer

# import sys
# import pathlib
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from GemR.exception import CustomException
# from dataclasses import dataclass



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logging.info(f"{filename} download! with following info: \n{headers}")
        else:
            logging.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  



    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir

        # creates a directory (folder) at the specified path unzip_path. If the directory already exists, the code does nothing.
        os.makedirs(unzip_path, exist_ok=True)

        # opens the zip file in read binary mode and extracts all files to the unzip_path directory.        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

