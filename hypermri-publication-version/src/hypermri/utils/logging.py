# Authors: Andre Wendlinger, andre.wendlinger@tum.de

# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023

import logging


LOG_MODES = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET,
}


def init_default_logger(
    name, default_mode="error", fstring="[%(levelname)s] %(name)s: %(message)s "
):
    """Initialize the default logger class.

    fstring:
        Format of the log message, example: '%(asctime)s:%(name)s:%(message)s '
    """
    logger = logging.getLogger(name)

    logger.setLevel(LOG_MODES[default_mode])

    formatter = logging.Formatter(fstring)
    # formatter = logging.Formatter()

    # Multiple handlers (to stream and/or to file) are possible.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # we have to ensure we don't add a handler every time the class is called.
    if not logger.hasHandlers():
        logger.addHandler(stream_handler)

    return logger
