# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------
import logging


class TrainerException(Exception):
    """
    Custom exception to use for custom validations.
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

    def __str__(self):
        if self.message:
            logging.error('TrainerException: ' + self.message)
            return 'TrainerException, {0} '.format(self.message)
        else:
            logging.warning('TrainerException has been raised')
            return 'TrainerException has been raised'
