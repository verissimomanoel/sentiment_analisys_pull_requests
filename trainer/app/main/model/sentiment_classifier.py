# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------

from app.main.model.abstract_classifier import AbstractClassifier
from app.main.model.config import Config


class SentimentAbstractClassifier(AbstractClassifier):
    """
    Sentiment classifier for sentiment analysis problems.
    """
    def __init__(self, config: Config) -> None:
        super(SentimentAbstractClassifier, self).__init__(config)
