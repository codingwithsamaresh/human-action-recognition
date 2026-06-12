from src.utils.logger import get_logger

logger = get_logger("test")

logger.info("Logger initialized successfully.")
logger.warning("This is a warning.")
logger.error("This is an error.")