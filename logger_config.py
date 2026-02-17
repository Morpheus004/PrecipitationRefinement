import logging
import os

def setup_logger(logger_name,log_file=None,level=None):
    """
    Configure and return a logger instance with both file and console handlers.
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    if level is None:
        logger.setLevel(logging.INFO)
    else: 
        logger.setLevel(level)
    #     return logger
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (this will be captured by sbatch redirection)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # # File handler (as backup)
    try:
        if log_file:
            log_file_path = log_file
        else:
            log_file_path = '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/model_training.log'

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        file_handler = logging.FileHandler(log_file_path,mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")

    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")
    #
    return logger

# Create a logger instance
