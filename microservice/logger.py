import os
import logging
from logging.handlers import TimedRotatingFileHandler


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with a specific name and configuration.
    """
    os.makedirs("logs", exist_ok=True)


    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = TimedRotatingFileHandler(
        filename="logs/app.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger