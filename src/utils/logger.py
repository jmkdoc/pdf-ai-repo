import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Function to set up a logger that logs to both console and file with rotating file handlers.

    :param name: Name of the logger.
    :param log_file: Path to the log file.
    :param level: Logging level.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = RotatingFileHandler(log_file, maxBytes=2000, backupCount=10)  # File handler

    # Create formatters and add them to the handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


# Example usage:
logger = setup_logger('my_logger', 'app.log')
logger.info('This is an info message.')
logger.error('This is an error message.')
