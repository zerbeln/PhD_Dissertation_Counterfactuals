import logging


def getLogger(name,
              level=logging.DEBUG,
              format='[%(levelname)s] [%(asctime)s] [%(name)s] - %(message)s',
              ):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # ch.terminator = ""

    # create formatter
    formatter = logging.Formatter(format)

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger

# good formats:
# format='%(asctime)s:%(name)s [%(levelname)s] - %(message)s',
