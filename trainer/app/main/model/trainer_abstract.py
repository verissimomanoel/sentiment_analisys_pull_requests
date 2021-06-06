# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------
import abc
import logging
import os

import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, AutoModelForSequenceClassification

from app.main.exception.trainer_exception import TrainerException
from app.main.model.config import Config
from app.main.utils.utils import preprocess, has_checkpoint


class TrainerAbstract(metaclass=abc.ABCMeta):
    """
    Default class for training
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        self.__validate_required_configs(config)
        self.__init_model(config)
        self.tokenizer = self.__init_tokenizer()

        self.df_train, self.df_test, self.df_val = self.get_data()
        self.train_data_loader = self.create_data_loader(self.df_train, self.tokenizer)
        self.val_data_loader = self.create_data_loader(self.df_val, self.tokenizer)
        self.test_data_loader = self.create_data_loader(self.df_test, self.tokenizer)

    def __init_model(self, config):
        """
        Init the model with baseline or checkpoint.
        :param config:
        :return:
        """
        if has_checkpoint(config.checkpoint_path):
            self.checkpoint = config.checkpoint_path
        else:
            self.checkpoint = config.baseline_path
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.config.device)

    def __validate_required_configs(self, config: Config) -> None:
        """
        Validate all required fields to run the experiment
        :param config: Configuration of experiment
        :return:
        """
        if config is None:
            raise TrainerException("Configurations is required")

        if config.device is None:
            raise TrainerException("Device configuration is required")

        if config.baseline_path is None or config.checkpoint_path is None:
            raise TrainerException("Baseline path and Checkpoint path configuration is required")

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
        df_val = pd.read_csv(self.config.val_path_file)
        df_test = pd.read_csv(self.config.test_path_file)

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
    def get_dataset(self, df: DataFrame, tokenizer: AutoTokenizer) -> Dataset:
        """
        Must return the specific class for dataset to training
        :param df:
        :param tokenizer:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval_model(self, data_loader: DataLoader, n_examples: int) -> tuple:
        """
        Must implement the rules to evaluate the model.
        :param data_loader:
        :param n_examples:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_predictions(self, data_loader) -> tuple:
        """
        Must implement get predictions of all data in a data loader.
        :param data_loader: The data loader to get the predictions
        :return:
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Execute the train loop.
        :return:
        """
        best_accuracy = 0
        early_stopping = self.config.early_stopping
        num_without_increase = 0
        optim = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in tqdm(range(self.config.epochs)):
            for batch in tqdm(self.train_data_loader, total=len(self.train_data_loader)):
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                targets = batch['targets'].to(self.config.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=targets)
                loss = outputs[0]
                loss.backward()
                optim.step()

            val_acc = self.eval_model(self.val_data_loader, len(self.df_val))

            logging.info("Accuracy Val: " + str(val_acc.item()) + "\n")

            if val_acc > best_accuracy:
                self.model.save_pretrained(self.config.checkpoint_path)
                best_accuracy = val_acc
            else:
                num_without_increase += 1

            if 0 < early_stopping < num_without_increase:
                logging.info("Early Stopping in " + str(epoch + 1))
                break

        self.eval()

    def eval(self) -> tuple:
        """
        Eval the model in a test data loader and export the results in result.csv file.
        :return:
        """
        test_acc = self.eval_model(self.test_data_loader, len(self.df_test))

        logging.info("Acc: " + str(test_acc.item()))

        y_texts, y_pred, y_pred_probs, y_test = self.get_predictions(self.test_data_loader)

        df_results = pd.DataFrame()
        df_results['y_texts'] = y_texts
        df_results['y_pred_probs'] = [t.numpy() for t in y_pred_probs]
        df_results['y_pred'] = y_pred

        if os.path.exists("/trainer/results/"):
            df_results.to_csv('/trainer/results/results.csv', index=False)
        else:
            directory = "../../results/"
            if not os.path.exists(directory):
                os.mkdir(directory)

            df_results.to_csv(directory + 'results.csv', index=False)
