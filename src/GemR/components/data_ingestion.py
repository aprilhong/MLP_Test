
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

    
    # def initiate_data_ingestion(self):
    #     logging.info('Enter the data ingestion method or component')
    #     try: 
    #         df=pd.read_csv('notebook\data\stud.csv') # change data location from db or sql as needed
    #         logging.info('Read the dataset as dataframe')

    #         os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

    #         df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

    #         logging.info('Train test split initiated')
    #         train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

    #         train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
    #         test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

    #         logging.info('Ingestion of data is completed')

    #         return(
    #             self.ingestion_config.train_data_path,
    #             self.ingestion_config.test_data_path,

    #         )

    #     except Exception as e:
    #         raise CustomException(e,sys)