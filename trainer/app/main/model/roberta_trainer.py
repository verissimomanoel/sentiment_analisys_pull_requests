# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------

import torch
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from app.main.model.config import Config
from app.main.model.sentiment_dataset import SentimentDataset
from app.main.model.trainer_abstract import TrainerAbstract


class RobertaTrainer(TrainerAbstract):
    """
    Class of training of model RoBERTa
    """

    def __init__(self, config: Config) -> None:
        super(RobertaTrainer, self).__init__(config)

    def get_dataset(self, df: DataFrame, tokenizer: AutoTokenizer) -> Dataset:
        ds = SentimentDataset(
            texts=df[self.config.feature_name].to_numpy(),
            targets=df[self.config.target_name].to_numpy(),
            tokenizer=tokenizer,
            max_len=self.config.max_len
        )

        return ds

    def eval_model(self, data_loader: DataLoader, n_examples: int) -> tuple:
        model = self.model.eval()

        correct_predictions = 0

        with torch.no_grad():
            for d in tqdm(data_loader, total=len(data_loader)):
                input_ids = d["input_ids"].to(self.config.device)
                attention_mask = d["attention_mask"].to(self.config.device)
                targets = d["targets"].to(self.config.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets,
                    return_dict=True
                )

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                correct_predictions += torch.sum(preds == targets)

        return correct_predictions.double() / n_examples

    def get_predictions(self, data_loader):
        model = self.model.eval()

        texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in tqdm(data_loader, total=len(data_loader)):
                text = d[self.config.feature_name]
                input_ids = d["input_ids"].to(self.config.device)
                attention_mask = d["attention_mask"].to(self.config.device)
                targets = d["targets"].to(self.config.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                    , return_dict=True)

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                texts.extend(text)
                predictions.extend(preds)
                prediction_probs.extend(logits)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()

        return texts, predictions, prediction_probs, real_values
