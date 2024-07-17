import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from GemR.logger import logging
from GemR.exception import CustomException
from GemR.entity.config_entity import DataTransformationConfig
from GemR.utils.common import save_object


@dataclass
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logging.info("Splited data into training and test sets")
        logging.info(train.shape)
        logging.info(test.shape)

        print(train.shape)
        print(test.shape)


    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = [
                "id", 
                "carat", 
                "depth", 
                "table", 
                "x", 
                "y", 
                "z"
                ]
            categorical_columns = [
                "cut",
                "color",
                "clarity"
            ]
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):

        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            logging.info("Read train and test data completed")

            train_x = train_data.drop([self.config.target_column], axis=1)
            test_x = test_data.drop([self.config.target_column], axis=1)
            train_y = train_data[[self.config.target_column]]
            test_y = test_data[[self.config.target_column]]

            logging.info("train and test data split complete")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()


            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr=preprocessing_obj.fit_transform(train_x)
            input_feature_test_arr=preprocessing_obj.transform(test_x)

            train_arr = np.c_[input_feature_train_arr, np.array(train_y)]
            test_arr = np.c_[input_feature_test_arr, np.array(test_y)]

            logging.info(f"Saved preprocessing object.")


            save_object(
                file_path=os.path.join(self.config.root_dir, "preprocessor.pkl"),
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.config.preprocessor
            )
        except Exception as e:
            raise CustomException(e,sys)
