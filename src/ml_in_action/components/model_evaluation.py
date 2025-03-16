import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from urllib.parse import urlparse
import numpy as np
import joblib

from ml_in_action.entity.config_entity import ModelEvaluationConfig
from ml_in_action.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def eval_metrics(self, actual, predicted):
        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(actual, predicted, average="weighted")
        recall = recall_score(actual, predicted, average="weighted")
        f1 = f1_score(actual, predicted, average="weighted")
        confusion = confusion_matrix(actual, predicted)

        return accuracy, precision, recall, f1, confusion

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop(columns=[self.config.target_column])
        test_y = test_data[self.config.target_column]

        predicted_labels = model.predict(test_x)

        # Compute classification metrics
        accuracy, precision, recall, f1, confusion = self.eval_metrics(
            test_y, predicted_labels
        )

        # Save metrics
        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": confusion.tolist(),  # Convert numpy array to list for JSON storage
        }
        save_json(path=Path(self.config.metric_file_name), data=scores)
