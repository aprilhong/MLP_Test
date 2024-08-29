import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler # OneHotEncoder
from sklearn.model_selection import train_test_split

from GemR.logger import logging
from GemR.exception import CustomException
from GemR.entity.config_entity import DataTransformationConfig
from GemR.utils.common import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config



    def data_cleaning(self):
        print(self.config.raw_data_path)
        data_raw = pd.read_csv(self.config.raw_data_path)
        
                
        # Copy dataframe and check the shape of the raw data
        print('Shape of dataframe:',data_raw.shape)

        # Check for duplicates
        df_row = len(data_raw)
        df_row_no_dupe = len(data_raw.drop_duplicates())
        df_row_dupe = df_row - df_row_no_dupe
        print('No. of rows with duplicates :', df_row_dupe)

        # Drop duplicates
        print('Shape of dataframe with duplicates dropped:', data_raw.drop_duplicates().shape)

        # Check for missing values 
        na_features=[features for features in data_raw.columns if data_raw[features].isnull().sum()>1]

        for feature in na_features:
            print ('features with missing_values')
            print(f"{feature:<15}{np.round(data_raw[feature].isnull().mean(), 4):^10}{'% missing values':>20}\n")

        #Drop missing values
        data = data_raw.dropna()

        #Check shape of orignal data and after dropping missing values
        print('Raw data shape:',data_raw.shape)
        print('Shape after dropping missing values:', data.shape,'\n')

        # Save cleaned data file
        data.to_csv(os.path.join(self.config.root_dir, "data.csv"),index = False)

        logging.info("Data cleaned and save as csv file")

    def train_test_spliting(self):
        
        # Load Data
        data = pd.read_csv(self.config.data_path)

        logging.info("Data Split Initiated")
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logging.info("Splitted data into training and test sets")
        print("Shape of training dataset",train.shape)
        print("Shape of testing dataset",test.shape)


    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())                
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                # ("one_hot_encoder",OneHotEncoder()),
                ('ordinal_encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ("scaler",StandardScaler())
                ]

            )

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_cols),
                ("cat_pipelines",cat_pipeline,categorical_cols)

                ]

            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):

        try:
            logging.info('Data transformation initiated')

            # Reading train and test data
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            print('train_df.head()\n',train_df.head(),'\n')
            print('test_df.head()\n',test_df.head(),'\n')

            # Get column names for both train and test dataframes, excluding the id column  
            train_header = train_df.columns.values[1:]
            test_header = test_df.columns.values[1:]


            logging.info('Read train and test data completed')
            print(f'Train Dataframe Head : \n{train_header}\n')
            print(f'Test Dataframe Head  : \n{test_header}\n')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()

            X_train_df = train_df.drop(columns=[self.config.target_column,'id'],axis=1)
            X_test_df=test_df.drop(columns=[self.config.target_column,'id'],axis=1)
            
            y_train_df= train_df[self.config.target_column]
            y_test_df=test_df[self.config.target_column]

            print("x_trained_head\n",X_train_df.head(),'\n')
            print("x_test_head\n",X_test_df.head(),'\n')
            print("y_trained_head\n",y_train_df.head(),'\n')
            print("y_test_head\n",y_test_df.head(),'\n')

            logging.info("Split X,y on training and testing datasets.")

            logging.info("Applying preprocessing object on X training and X testing datasets.")
            
            X_train_arr=preprocessing_obj.fit_transform(X_train_df)
            X_test_arr=preprocessing_obj.transform(X_test_df)
            

            logging.info("Preprocessing object APPLIED on X training and X testing dataframe.")

            print(f"Shape of X_train_arr{X_train_arr.shape}")
            print(f"Shape of X_test_arr{X_test_arr.shape}")

            # Concat X and y on training and testing datasets
            train_arr = np.c_[X_train_arr, np.array(y_train_df)]
            test_arr = np.c_[X_test_arr, np.array(y_test_df)]

            logging.info("X and Y column merged after preprocessing is complete.")


            # Save merged training and testing data to a CSV file
            np.savetxt(os.path.join(self.config.root_dir,"train_array.csv"), train_arr, delimiter=',', header=','.join(train_header))
            np.savetxt(os.path.join(self.config.root_dir,"test_array.csv"), test_arr, delimiter=',',header=','.join(test_header))
            logging.info("Preprocessed Test and Train data saved to .csv file")

            
            save_object(
                file_path=os.path.join(self.config.root_dir, "preprocessor.pkl"),
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object.")


            return (
                train_arr,
                test_arr,
                self.config.preprocessor
            )
        

        except Exception as e:
            raise CustomException(e,sys)
