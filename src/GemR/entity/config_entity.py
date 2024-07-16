from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    # train_data_path: str=os.path.join('artifacts','train.csv')
    # test_data_path: str=os.path.join('artifacts','test.csv')
    # raw_data_path: str=os.path.join('artifacts','data.csv')

# @dataclass(frozen=True)
# class DataIngestionConfig:
#     train_data_path: str=os.path.join('artifacts','train.csv')
#     test_data_path: str=os.path.join('artifacts','test.csv')
#     raw_data_path: str=os.path.join('artifacts','data.csv')


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


#Option for mlflow
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    alpha: float
    l1_ratio: float
    target_column: str

# # Option 2
# @dataclass(frozen=True)
# class ModelTrainerConfig: 
#     trained_model_file_path=os.path.join('artifacts','model.pkl')

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str