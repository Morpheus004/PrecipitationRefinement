import logging

def setup_logger():
    """
    Configure and return a logger instance.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Create a logger instance
logger = setup_logger() 