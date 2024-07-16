import sys
from GemR.logger import logging

def error_message_detail(error,error_detail:sys):
    """
    Creates a detailed error message with information about the error and its location.

    Args:
        error (str): The original error message.
        error_detail (sys, optional): An object containing exception details (usually from `sys.exc_info()`).
            Defaults to None.

    Returns:
        str: A formatted error message with filename, line number, and original error message.
    """

    # Unpack the exception information if provided
    __,__,exc_tb = error_detail.exc_info()

    # Extract filename and line number from the traceback
    file_name=exc_tb.tb_frame.f_code.co_filename

    # Format the error message with details
    error_message='Error occured in python script name [{0}] line number [{1}] error message [{2}]'.format(
        file_name,exc_tb.tb_lineno,str(error))
    
    return error_message
    

class CustomException(Exception):
    """
    Custom exception class for handling errors with detailed messages.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the custom exception with an error message and optional details.

        Args:
            error_message (str): The main error message to be displayed.
            error_detail (sys, optional): Additional details about the error. Defaults to sys.exc_info().
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  

    def __str__(self):
        """Returns the error message for string representation."""
        return self.error_message


# # Test if exception.py works
if __name__=='__main__':
    try:
        a=1/0
    except Exception as e:
            logging.info('Divide by Zero')
            raise CustomException(e,sys)