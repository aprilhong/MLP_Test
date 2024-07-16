import logging
import os
import sys
from datetime import datetime

# Define the log file name format (using datetime for unique filenames)
LOG_FILE = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'

# Construct the logs directory path (create it if it doesn't exist)
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(logs_path, exist_ok=True)  # Create directories if they don't exist

# Combine the logs directory and log file name for full path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Custom log message format
    level=logging.INFO,  # Set the minimum logging level (INFO in this case)

    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # create log folder and files
        logging.StreamHandler(sys.stdout)   # print log message in terminal
    ]    
)

# run to test if logger works
if __name__=='__main__':
    logging.info('Logging has started')