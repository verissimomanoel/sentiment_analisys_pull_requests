# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------
import click
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.main.api.sentiment_service import SentimentService


@click.command()
@click.option("--path-of-model", type=click.STRING, default="../volume/model")
def publish(path_of_model):
    """
    Publish the model on BentoML and print the saved path.
    :param path_of_model:
    :return:
    """
    ts = SentimentService()
    model = AutoModelForSequenceClassification.from_pretrained(path_of_model)
    tokenizer = AutoTokenizer.from_pretrained(path_of_model)
    artifact = {"model": model, "tokenizer": tokenizer}
    ts.pack("sentimentModel", artifact)
    saved_path = ts.save()
    print(saved_path)

if __name__ == '__main__':
    publish()