import sys
from GemR.logger import logging
from GemR.exception import CustomException
from GemR.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from GemR.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from GemR.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from GemR.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from GemR.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


# Run to test logger.py
# logging.info("Welcome to my custom classifier!")

# Run to test exception.py
# try:
#     a=1/0
# except Exception as e:
#         logging.info('Divide by Zero')
#         raise CustomException(e,sys)

# Run DataIngestionTrainingPipeline

# STAGE_NAME = "Data Ingestion Stage"

# try:
#     logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     data_ingestion = DataIngestionTrainingPipeline()
#     data_ingestion.main()
#     logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     raise CustomException(e,sys)
    

# STAGE_NAME = "Data Validation stage"

# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataValidationTrainingPipeline()
#    data_ingestion.main()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     raise CustomException(e,sys)



# STAGE_NAME = "Data Transformation stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataTransformationTrainingPipeline()
#    data_ingestion.main()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     raise CustomException(e,sys)




# STAGE_NAME = "Model Trainer stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = ModelTrainerTrainingPipeline()
#    data_ingestion.main()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     raise CustomException(e,sys)


STAGE_NAME = "Model evaluation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelEvaluationTrainingPipeline()
   data_ingestion.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e,sys)






