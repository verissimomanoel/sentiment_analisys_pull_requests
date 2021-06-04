# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------
from typing import Type

import numpy as np
import torch
import torch.nn.functional as F
from pandas import DataFrame
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AdamW

from app.main.model.abstract_classifier import AbstractClassifier
from app.main.model.config import Config
from app.main.model.sentiment_dataset import SentimentDataset
from app.main.model.trainer_abstract import TrainerAbstract


class RobertaTrainer(TrainerAbstract):
    """
    Class of training of model RoBERTa
    """
    def __init__(self, config: Config, model: Type[AbstractClassifier]) -> None:
        super(RobertaTrainer, self).__init__(config, model)

    def train_epoch(self, model, data_loader, loss_fn, optimizer, device, scheduler, n_examples) -> tuple:
        model = model.train()

        losses = []
        correct_predictions = 0

        for d in tqdm(data_loader, total=len(data_loader)):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)

    def get_dataset(self, df: DataFrame, tokenizer: AutoTokenizer) -> Dataset:
        ds = SentimentDataset(
            texts=df[self.config.feature_name].to_numpy(),
            targets=df[self.config.target_name].to_numpy(),
            tokenizer=tokenizer,
            max_len=self.config.max_len
        )

        return ds

    def configure_train(self) -> tuple:
        data = next(iter(self.train_data_loader))

        model = self.model.to(self.config.device)

        input_ids = data['input_ids'].to(self.config.device)
        attention_mask = data['attention_mask'].to(self.config.device)

        F.softmax(model(input_ids, attention_mask), dim=1)

        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, correct_bias=False)
        total_steps = len(self.train_data_loader) * self.config.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss().to(self.config.device)

        return loss_fn, scheduler, optimizer

    def eval_model(self, data_loader, loss_fn, n_examples) -> tuple:
        model = self.model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in tqdm(data_loader, total=len(data_loader)):
                input_ids = d["input_ids"].to(self.config.device)
                attention_mask = d["attention_mask"].to(self.config.device)
                targets = d["targets"].to(self.config.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, targets)

                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)