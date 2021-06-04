# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------
import os

from torch import nn
from transformers import AutoModel

from app.main.exception.trainer_exception import TrainerException
from app.main.model.config import Config


class AbstractClassifier(nn.Module):
    """
    Default class for classifiers.
    """

    def __init__(self, config: Config) -> None:
        super(AbstractClassifier, self).__init__()
        self.__validate_required_configs(config)
        if config.checkpoint_path and len(os.listdir(config.checkpoint_path)) > 0:
            self.checkpoint = config.checkpoint_path
        else:
            self.checkpoint = config.baseline_path

        self.bert = AutoModel.from_pretrained(self.checkpoint)
        self.drop = nn.Dropout(p=config.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, config.n_classes)

    def __validate_required_configs(self, config: Config) -> None:
        """
        Validate all required fields to run the experiment
        :param config: Configuration of experiment
        :return:
        """
        if config is None:
            raise TrainerException("Configurations is required")

        if config.baseline_path is None or config.checkpoint_path is None:
            raise TrainerException("Baseline path and Checkpoint path configuration is required")

        if config.n_classes is None:
            raise TrainerException("Number of classes configuration is required")

    def forward(self, input_ids, attention_mask):
        """
        Execute the forward phase for training neural networks.
        :param input_ids: The input ids
        :param attention_mask: The masks for attention
        :return:
        """
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = self.drop(pooled_output)
        return self.out(output)

