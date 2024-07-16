import sys
from GemR.config.configuration import ConfigurationManager
from GemR.components.data_validation import DataValidation
from GemR.logger import logging
from GemR.exception import CustomException

STAGE_NAME = "Data Validation Stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()

# Initiate Pipeline
if __name__ == '__main__':
    try: 
        logging.info(f'>>>>>> stage {STAGE_NAME} started <<<<<<')
        obj = DataValidationTrainingPipeline()
        obj.main()
        logging.info(f'>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n x============x')
    except Exception as e:
        raise CustomException(e, sys)
