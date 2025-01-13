from loguru import logger

def setup_logger():
    logger.add("../logs/general.log", rotation="500 MB")
    logger.info("Logger is set up.")