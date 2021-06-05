# ----------------------------------------------------------------------------------------------------------------------
#   This software is free software.
# ----------------------------------------------------------------------------------------------------------------------


class Config:
    """
    Class with configurations of training models.
    """
    device: str
    n_classes: int
    early_stopping: int = 0
    epochs: int = 10
    train_path_file: str
    val_path_file: str
    test_path_file: str
    max_len: int = 256
    batch_size: int = 18
    feature_name: str
    target_name: str
    num_workers: int = 4
    baseline_path: str = "baseline"
    checkpoint_path: str
    learning_rate: float = 2e-5

