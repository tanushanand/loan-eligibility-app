"""
Logger configuration for the Loan Eligibility App.
"""

import logging

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger