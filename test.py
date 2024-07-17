import os
import sys
import urllib.request as request
from urllib.parse import urlparse
import zipfile
import pandas as pd
from GemR.logger import logging
from sklearn.model_selection import train_test_split
from GemR.exception import CustomException
from dataclasses import dataclass
from GemR.utils.common import get_size
from GemR.entity.config_entity import (DataIngestionConfig)
from pathlib import Path
from GemR.components.data_transformation import DataTransformation, DataTransformationConfig
from GemR.components.model_trainer import ModelTrainerConfig, ModelTrainer
import pathlib

