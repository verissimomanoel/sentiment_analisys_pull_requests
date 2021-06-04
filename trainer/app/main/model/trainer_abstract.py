# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------
import abc
import logging
import os
from collections import defaultdict
from typing import Type

import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from app.main.exception.trainer_exception import TrainerException
from app.main.model.abstract_classifier import AbstractClassifier
from app.main.model.config import Config
from app.main.utils.utils import preprocess


class TrainerAbstract(metaclass=abc.ABCMeta):
    """
    Default class for training
    """

    def __init__(self, config: Config, model: Type[AbstractClassifier]) -> None:
        self.config = config
        self.__validate_required_configs(config, model)
        self.model = model
        self.df_train, self.df_test, self.df_val = self.get_data()

        self.tokenizer = self.__init_tokenizer()

        self.train_data_loader = self.create_data_loader(self.df_train, self.tokenizer)
        self.val_data_loader = self.create_data_loader(self.df_val, self.tokenizer)
        self.test_data_loader = self.create_data_loader(self.df_test, self.tokenizer)

    def __validate_required_configs(self, config: Config, model: Type[AbstractClassifier]) -> None:
        """
        Validate all required fields to run the experiment
        :param config: Configuration of experiment
        :param model: Model to run the training
        :return:
        """
        if config is None:
            raise TrainerException("Configurations is required")

        if model is None:
            raise TrainerException("Model is required")

        if config.device is None:
            raise TrainerException("Device configuration is required")

        if config.baseline_path is None or config.checkpoint_path is None:
            raise TrainerException("Baseline path or Checkpoint path configuration is required")

        if config.train_path_file is None:
            raise TrainerException("Train path file configuration is required")

        if config.val_path_file is None:
            raise TrainerException("Validation path file configuration is required")

        if config.test_path_file is None:
            raise TrainerException("Test path file configuration is required")

        if config.n_classes is None:
            raise TrainerException("Number of classes configuration is required")

        if config.feature_name is None:
            raise TrainerException("Feature name configuration is required")

        if config.target_name is None:
            raise TrainerException("Target name configuration is required")

    def __init_tokenizer(self) -> AutoTokenizer:
        """
        Init the tokenizer and save in checkpoint path.
        :return: The Tokenizer
        """
        if self.config.checkpoint_path and len(os.listdir(self.config.checkpoint_path)) > 0:
            tokenizer_path = self.config.checkpoint_path
        else:
            tokenizer_path = self.config.baseline_path

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(self.config.checkpoint_path);

        return tokenizer

    def get_data(self) -> tuple:
        """
        Get the data for training, validation and test
        :return:
        """
        df_train = pd.read_csv(self.config.train_path_file)
        df_train = df_train[:100]
        df_val = pd.read_csv(self.config.val_path_file)
        df_val = df_val[:20]
        df_test = pd.read_csv(self.config.test_path_file)
        df_test = df_test[:20]

        df_train = preprocess(df_train)
        df_val = preprocess(df_val)
        df_test = preprocess(df_test)

        return df_train, df_test, df_val

    def create_data_loader(self, df: DataFrame, tokenizer: AutoTokenizer) -> DataLoader:
        """
        Create the data loader to training
        :param df: Data frame to create the dataloader
        :param tokenizer: Tokenizer of texts
        :return:
        """
        ds = self.get_dataset(df, tokenizer)

        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )

    @abc.abstractmethod
    def train_epoch(self, model, data_loader, loss_fn, optimizer, device, scheduler, n_examples) -> tuple:
        """
        Implements the training per each epoch.
        :param model: The model for training
        :param data_loader: The data loader
        :param loss_fn: The function of loss
        :param optimizer: The optimizer of model
        :param device: The device for training (CPU or GPU)
        :param scheduler: The scheduler for change the leaning rate
        :param n_examples: The number of examples for training
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def configure_train(self) -> tuple:
        """
        Must configure parameters like: Loss Function, Scheduler for learning rate adjusts
        and optimizer for training.
        :return: Tuple with: loss_fn, scheduler, optimizer
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset(self, df: DataFrame, tokenizer: AutoTokenizer) -> Dataset:
        """
        Must return the specific class for dataset to training
        :param df:
        :param tokenizer:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval_model(self, data_loader, loss_fn, n_examples) -> tuple:
        """
        Must implement the rules to evaluate the model.
        :param data_loader:
        :param loss_fn:
        :param n_examples:
        :return:
        """

    def train(self) -> None:
        """
        Execute the train loop.
        :return:
        """
        loss_fn, scheduler, optimizer = self.configure_train()

        history = defaultdict(list)
        best_accuracy = 0
        early_stopping = self.config.early_stopping
        num_without_increase = 0

        for epoch in tqdm(range(self.config.epochs)):
            logging.info("Epoch " + str(epoch + 1) + " - " + str(self.config.epochs))

            train_acc, train_loss = self.train_epoch(
                self.model,
                self.train_data_loader,
                loss_fn,
                optimizer,
                self.config.device,
                scheduler,
                len(self.df_train)
            )

            logging.info("Train loss " + str(train_loss) + " accuracy " + str(train_acc))

            val_acc, val_loss = self.eval_model(self.val_data_loader, loss_fn, len(self.df_val))

            logging.info("Val loss " + str(val_loss) + " accuracy " + str(val_acc) + "\n")

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                self.model.bert.save_pretrained(self.config.checkpoint_path)
                best_accuracy = val_acc
            else:
                num_without_increase += 1

            if 0 < early_stopping < num_without_increase:
                logging.info("Early Stopping in " + str(epoch))
                break
