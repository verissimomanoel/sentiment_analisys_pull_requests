# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------
import re
import string
from pathlib import Path

import nltk
from pandas import DataFrame


def preprocess(data) -> DataFrame:
    """
    Pre process the data removing hashtags, numbers, punctuation, links and stopwords.
    :param data: Dataframe with the data.
    :return:
    """
    nltk.download('stopwords')
    stopwords = list(set([w for w in nltk.corpus.stopwords.words("english")]))

    """Remove hashtaghs, numbers, punctuation, links and stopwords."""
    df = data

    df["text"] = df["text"].apply(lambda s: s.lower())
    df["text"] = df["text"].apply(lambda x: re.sub("#[^ ]+", "", x))  # remove hashtags
    df["text"] = df["text"].apply(lambda x: re.sub("\d+", "", x))  # remove numbers
    df["text"] = df["text"].apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation)))  # remove punctuation
    df["text"] = df["text"].apply(lambda x: re.sub("http[^ ]+", "", x))  # remove links
    df["text"] = df["text"].apply(
        lambda x: " ".join(w.strip() for w in x.split() if w not in stopwords))  # remove stopword

    return df


def has_checkpoint(checkpoint_path):
    """
    Check the checkpoint exists
    :param checkpoint_path:
    :return:
    """
    return checkpoint_path and Path(checkpoint_path + "/pytorch_model.bin").is_file()
