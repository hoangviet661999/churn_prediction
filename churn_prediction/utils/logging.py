import functools
import logging
import sys

logger_initialized = {}


@functools.lru_cache()
def get_logger(name: str = "mlops", log_file: str = None, log_level=logging.DEBUG):
    """
    Initialize a logger by name.
    If the logger hasn't been initialized, the logger will initialize a StreamHandler
    by deafault and if the log_file was specified, a FileHandler will be initialized,
    otherwise the logger will
    be returned directly.

    Parameters:
        name(str): name of logger.
        log_file(str | None): logged file by FileHandler.
        log_level(int): level to log.

    Returns:
        logger(logging.Logger): the expected logger.
    """

    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger

    formatter = logging.Formatter(
        "%(levelname)s:\t %(filename)s:%(lineno)d - %(message)s"
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(log_level)

    return logger
