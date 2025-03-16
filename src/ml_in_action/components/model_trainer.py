import pandas as pd
import os

from sklearn.neighbors import KNeighborsClassifier
from ml_in_action import logger
import joblib

from ml_in_action.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        model = KNeighborsClassifier(
            n_neighbors=self.config.n_neighbors,
            weights=self.config.weights,
            p=self.config.p,
        )
        model.fit(train_x, train_y)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
