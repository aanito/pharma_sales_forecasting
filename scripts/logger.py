from loguru import logger
import os

def setup_logger():
    """Sets up the logger configuration."""
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)  # Create the logs directory if it doesn't exist
    
    logger.remove()  # Remove the default logger

    logger.add(f"{log_directory}/data_preprocessing.log", rotation="500 MB", 
               level="INFO", format="{time} {level} {message}", 
               backtrace=True, diagnose=True)
    
    logger.add(f"{log_directory}/competitor_distance_effect.log", rotation="500 MB", 
               level="INFO", format="{time} {level} {message}", 
               backtrace=True, diagnose=True)
    
    logger.add(f"{log_directory}/exploratory_analysis.log", rotation="500 MB", 
               level="INFO", format="{time} {level} {message}", 
               backtrace=True, diagnose=True)
    
    # logger.add(f"{log_directory}/competitor_distance_effect.log", rotation="500 MB", 
    #            level="INFO", format="{time} {level} {message}", 
    #            backtrace=True, diagnose=True)
    
    # logger.add(f"{log_directory}/competitor_distance_effect.log", rotation="500 MB", 
    #            level="INFO", format="{time} {level} {message}", 
    #            backtrace=True, diagnose=True)
    
    # logger.add(f"{log_directory}/competitor_distance_effect.log", rotation="500 MB", 
    #            level="INFO", format="{time} {level} {message}", 
    #            backtrace=True, diagnose=True)
    
    # logger.add(f"{log_directory}/competitor_distance_effect.log", rotation="500 MB", 
    #            level="INFO", format="{time} {level} {message}", 
    #            backtrace=True, diagnose=True)
    
    # logger.add(f"{log_directory}/competitor_distance_effect.log", rotation="500 MB", 
    #            level="INFO", format="{time} {level} {message}", 
    #            backtrace=True, diagnose=True)
    
    # logger.add(f"{log_directory}/competitor_distance_effect.log", rotation="500 MB", 
    #            level="INFO", format="{time} {level} {message}", 
    #            backtrace=True, diagnose=True)
    
    # logger.add(f"{log_directory}/competitor_distance_effect.log", rotation="500 MB", 
    #            level="INFO", format="{time} {level} {message}", 
    #            backtrace=True, diagnose=True)
    
    
    logger.info("Logger is set up.")
