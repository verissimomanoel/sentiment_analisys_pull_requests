# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------
import bentoml
import numpy as np
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.types import InferenceError
from schema import Schema, SchemaError
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk
import re
import string


@bentoml.env(pip_packages=["transformers==3.5.1", "torch==1.5.0", "pandas",
                           "nltk==3.6.2", "scipy==1.5.2", "schema==0.7.4"])
@bentoml.artifacts([TransformersModelArtifact("sentimentModel")])
class SentimentService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        """
        Realize the prediction of the input json.
        :param parsed_json: JSON in a format: [{'message': str, "meta": {"commit_id": str, "comment_id": str}}]
        :return:
        """
        try:
            self.__validate_schema(parsed_json)
        except SchemaError as err:
            return InferenceError(err_msg="Invalid json input:" + err.code, http_status=400)

        model = self.artifacts.sentimentModel.get("model")
        tokenizer = self.artifacts.sentimentModel.get("tokenizer")

        output_list = self.__process(model, tokenizer, parsed_json)

        return output_list

    def __process(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, parsed_json: dict) -> dict:
        """
        Proess the input and generate the output prediction.
        :param model:
        :param tokenizer:
        :param parsed_json:
        :return:
        """
        labels = ["Negative", "Neutral", "Positive"]
        output_list = []
        for obj in parsed_json:
            test_sentence = self.__preprocess(obj["message"])
            inputs = tokenizer(test_sentence, return_tensors="pt")
            outputs = model(**inputs)
            scores = outputs[0][0].detach().numpy()
            scores = softmax(scores)

            ranking = np.argsort(scores)
            ranking = ranking[::-1]

            output = {}
            for i in range(scores.shape[0]):
                label = labels[ranking[i]]
                score = scores[ranking[i]]
                output[label.lower() + "_score"] = np.round(float(score), 2)

            output["meta"] = obj["meta"]
            output_list.append(output)

        return output_list

    def __preprocess(self, text: str) -> str:
        """
        Remove hashtaghs, numbers, punctuation, links and stopwords.
        :param text:
        :return:
        """
        nltk.download('stopwords')
        stopwords = list(set([w for w in nltk.corpus.stopwords.words("english")]))

        text = text.lower()
        text = re.sub("#[^ ]+", "", text)  # remove hashtags
        text = re.sub("\d+", "", text)  # remove numbers
        text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
        text = re.sub("http[^ ]+", "", text)  # remove links
        text = " ".join(w.strip() for w in text.split() if w not in stopwords)  # remove stopword

        return text

    def __validate_schema(self, parsed_json) -> None:
        """
        Validate the json input with the valid schema.
        :param parsed_json:
        :return:
        """
        schema = Schema([{'message': str, "meta": {"commit_id": str, "comment_id": str}}])
        schema.validate(parsed_json)