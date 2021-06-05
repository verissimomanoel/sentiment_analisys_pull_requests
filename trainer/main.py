# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------

import click
import torch

from app.main.model.config import Config
from app.main.model.roberta_trainer import RobertaTrainer


@click.command()
@click.option("--number-of-classes", type=click.INT)
@click.option("--early-stopping", type=click.INT, default=3)
@click.option("--epochs", type=click.INT, default=10)
@click.option("--train-path-file", type=click.Path(exists=True))
@click.option("--val-path-file", type=click.Path(exists=True))
@click.option("--test-path-file", type=click.Path(exists=True))
@click.option("--max-len", type=click.INT, default=256)
@click.option("--batch-size", type=int, default=24)
@click.option("--feature-name", type=click.STRING)
@click.option("--target-name", type=click.STRING)
@click.option("--num-workers", type=click.INT, default=4)
@click.option("--baseline-path", type=click.Path(exists=True), default="baseline")
@click.option("--checkpoint-path", type=click.Path(exists=True))
@click.option("--learning-rate", type=click.FLOAT, default=2e-5)
def main(number_of_classes: int, early_stopping: int, epochs: int, train_path_file: str,
         val_path_file: str, test_path_file: str, max_len: int, batch_size: int, feature_name: str,
         target_name: str, num_workers: int, baseline_path: str, checkpoint_path: str,
         learning_rate: int):
    config = Config()
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.n_classes = number_of_classes
    config.early_stopping = early_stopping
    config.epochs = epochs
    config.train_path_file = train_path_file
    config.val_path_file = val_path_file
    config.test_path_file = test_path_file
    config.max_len = max_len
    config.batch_size = batch_size
    config.feature_name = feature_name
    config.target_name = target_name
    config.num_workers = num_workers
    config.baseline_path = baseline_path
    config.checkpoint_path = checkpoint_path
    config.learning_rate = learning_rate

    roberta_trainer = RobertaTrainer(config)
    roberta_trainer.train()

if __name__ == '__main__':
    main()
