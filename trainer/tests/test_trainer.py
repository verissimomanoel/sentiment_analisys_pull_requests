import unittest

import torch

from app.main.model.config import Config
from app.main.model.roberta_trainer import RobertaTrainer
from app.main.model.sentiment_classifier import SentimentAbstractClassifier


class TrainerTestCase(unittest.TestCase):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "../baseline"
    TRAIN_PATH_FILE = "../data/train.csv"
    VAL_PATH_FILE = "../data/val.csv"
    def test_without_config(self):
        with self.assertRaises(Exception) as context:
            SentimentAbstractClassifier(None)

        self.assertTrue('Configurations is required' in context.exception.message)

    def test_without_baseline_path(self):
        config = Config()
        config.checkpoint_path = None
        with self.assertRaises(Exception) as context:
            SentimentAbstractClassifier(config)

        self.assertTrue('Baseline path and Checkpoint path configuration is required' in context.exception.message)

    def test_without_number_of_classes(self):
        config = Config()
        config.checkpoint_path = "./baseline"
        config.n_classes = None
        with self.assertRaises(Exception) as context:
            SentimentAbstractClassifier(config)

        self.assertTrue('Number of classes configuration is required' in context.exception.message)

    def test_without_device(self):
        config = Config()
        config.checkpoint_path = self.CHECKPOINT_PATH
        config.n_classes = 3
        config.test_path_file = None
        config.device = None
        with self.assertRaises(Exception) as context:
            model = SentimentAbstractClassifier(config)
            RobertaTrainer(config, model)

        self.assertTrue('Device configuration is required' in context.exception.message)

    def test_without_train_path_file(self):
        config = Config()
        config.device = self.DEVICE
        config.checkpoint_path = self.CHECKPOINT_PATH
        config.n_classes = 3
        config.train_path_file = None
        with self.assertRaises(Exception) as context:
            model = SentimentAbstractClassifier(config)
            RobertaTrainer(config, model)

        self.assertTrue('Train path file configuration is required' in context.exception.message)

    def test_without_val_path_file(self):
        config = Config()
        config.device = self.DEVICE
        config.checkpoint_path = self.CHECKPOINT_PATH
        config.n_classes = 3
        config.train_path_file = self.TRAIN_PATH_FILE
        config.val_path_file = None
        with self.assertRaises(Exception) as context:
            model = SentimentAbstractClassifier(config)
            RobertaTrainer(config, model)

        self.assertTrue('Validation path file configuration is required' in context.exception.message)

    def test_without_test_path_file(self):
        config = Config()
        config.device = self.DEVICE
        config.checkpoint_path = self.CHECKPOINT_PATH
        config.n_classes = 3
        config.train_path_file = self.TRAIN_PATH_FILE
        config.val_path_file = self.VAL_PATH_FILE
        config.test_path_file = None
        with self.assertRaises(Exception) as context:
            model = SentimentAbstractClassifier(config)
            RobertaTrainer(config, model)

        self.assertTrue('Test path file configuration is required' in context.exception.message)

    def test_without_feature_name(self):
        config = Config()
        config.device = self.DEVICE
        config.checkpoint_path = self.CHECKPOINT_PATH
        config.n_classes = 3
        config.train_path_file = self.TRAIN_PATH_FILE
        config.val_path_file = self.VAL_PATH_FILE
        config.test_path_file = "../data/test.csv"
        config.feature_name = None
        with self.assertRaises(Exception) as context:
            model = SentimentAbstractClassifier(config)
            RobertaTrainer(config, model)

        self.assertTrue('Feature name configuration is required' in context.exception.message)

    def test_without_target_name(self):
        config = Config()
        config.device = self.DEVICE
        config.checkpoint_path = self.CHECKPOINT_PATH
        config.n_classes = 3
        config.train_path_file = self.TRAIN_PATH_FILE
        config.val_path_file = self.VAL_PATH_FILE
        config.test_path_file = "../data/test.csv"
        config.feature_name = "text"
        config.target_name = None
        with self.assertRaises(Exception) as context:
            model = SentimentAbstractClassifier(config)
            RobertaTrainer(config, model)

        self.assertTrue('Target name configuration is required' in context.exception.message)

if __name__ == '__main__':
    unittest.main()
