import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
# import mlflow
from sklearn.model_selection import GridSearchCV
from GemR.exception import CustomException
from GemR.logger import logging
import numpy as np
# import joblib
from GemR.entity.config_entity import ModelEvaluationConfig
from GemR.utils.common import save_json, load_object



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def predict(self, features):

        try:
            model = load_object(file_path=self.config.model_path)
            custom_data = 
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
               
#     def eval_metrics(self,actual, pred):
#         rmse = np.sqrt(mean_squared_error(actual, pred))
#         mae = mean_absolute_error(actual, pred)
#         r2 = r2_score(actual, pred)
#         return rmse, mae, r2
    
    
# def evaluate_models(X_train, y_train, X_test, y_test, models,param):
#     try:
#         report = {}

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para=param[list(models.keys())[i]]

#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,y_train)

#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)

#             #model.fit(X_train, y_train)  # Train model

#             y_train_pred = model.predict(X_train)

#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train, y_train_pred)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)
    

    # def log_into_mlflow(self):

    #     test_data = pd.read_csv(self.config.test_data_path)
    #     model = joblib.load(self.config.model_path)

    #     test_x = test_data.drop([self.config.target_column], axis=1)
    #     test_y = test_data[[self.config.target_column]]


    #     mlflow.set_registry_uri(self.config.mlflow_uri)
    #     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


    #     with mlflow.start_run():

    #         predicted_qualities = model.predict(test_x)

    #         (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
    #         # Saving metrics as local
    #         scores = {"rmse": rmse, "mae": mae, "r2": r2}
    #         save_json(path=Path(self.config.metric_file_name), data=scores)

    #         mlflow.log_params(self.config.all_params)

    #         mlflow.log_metric("rmse", rmse)
    #         mlflow.log_metric("r2", r2)
    #         mlflow.log_metric("mae", mae)


    #         # Model registry does not work with file store
    #         if tracking_url_type_store != "file":

    #             # Register the model
    #             # There are other ways to use the Model Registry, which depends on the use case,
    #             # please refer to the doc for more information:
    #             # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    #             mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
    #         else:
    #             mlflow.sklearn.log_model(model, "model")

    
