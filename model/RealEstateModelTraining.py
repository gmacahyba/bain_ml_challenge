import pandas as pd
import numpy as np
import os
from typing import Callable
import pickle

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error)

from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
import logging

logging.basicConfig(filename = "./model/model_training.log", 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
logger = logging.getLogger(__name__)

    
class CSVLoader:
    def __init__(self, train_path: str = None, test_path: str = None) -> 'CSVLoader':
        if train_path is None:
            train_path = './model/train.csv'
        if test_path is None:
            test_path = './model/test.csv'
        self.train_path = train_path
        self.test_path = test_path
    
    def load_csv_data(self) -> (pd.DataFrame, pd.DataFrame):
        return pd.read_csv(self.train_path), pd.read_csv(self.test_path)

class RealEstateModel:
    def __init__(self, categorical_transformer: Callable | None = None, categorical_cols: list = None, preprocessor = None, steps: list = None) -> 'RealEstateModel':
        if categorical_transformer is None:
            categorical_transformer = TargetEncoder()
        if categorical_cols is None:
            categorical_cols = ['type', 'sector']
        self.categorical_transformer = categorical_transformer
        self.categorical_cols = categorical_cols
        if preprocessor is None:
            preprocessor = ColumnTransformer(transformers=[
                        ('categorical',
                        self.categorical_transformer,
                        self.categorical_cols)
                    ])
        self.preprocessor = preprocessor
        if steps is None:
            steps = [
                ('preprocessor', self.preprocessor),
                ('model', GradientBoostingRegressor(**{
                    "learning_rate":0.01,
                    "n_estimators":300,
                    "max_depth":5,
                    "loss":"absolute_error"
            }))
        ]
        self.steps = steps
        self.pipeline = Pipeline(self.steps)
        self.train_cols = ['type', 'sector','net_usable_area', 'net_area', 'n_rooms', 'n_bathroom', 'latitude', 'longitude']

    def train(self, real_estate: pd.DataFrame) -> None:
        targets = real_estate['price']
        self.pipeline.fit(real_estate[self.train_cols], targets)

    def estimate(self, real_estate: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(real_estate[self.train_cols])

class ModelSaver:
    def __init__(self, model_filename: str, result_filename: str) -> 'ModelSaver':
        self.model_filename = model_filename
        self.result_filename = result_filename

    def save_model(self, model, result):
        pickle.dump(model, open(self.model_filename, 'wb'))
        pickle.dump(result, open(self.result_filename, 'wb'))

class ModelMetrics:
    def __init__(self):
        pass

    def compute_metrics(self, predictions: list, target: list) -> list:
        return [np.sqrt(mean_squared_error(predictions, target)), mean_absolute_percentage_error(predictions, target), mean_absolute_error(predictions, target)]
    
class RealEstateModelCreator:
    def __init__(self, loader: CSVLoader, model: RealEstateModel, model_saver: ModelSaver, model_metrics: ModelMetrics) -> 'RealEstateModelCreator':
        self.loader = loader
        self.model = model
        self.model_saver = model_saver
        self.model_metrics = model_metrics

    def run(self):
        real_estate_train, real_estate_test = self.loader.load_csv_data()

        # --- TRAINING ---
        logger.info("Starting model training...")
        self.model.train(real_estate_train)
        logger.info("Training complete!")
        y_train_estimation = self.model.estimate(real_estate_train)
        metrics_train = self.model_metrics.compute_metrics(
             y_train_estimation, real_estate_train['price']
        )
        logger.info(f"Computed metrics (train): {metrics_train}")
        
        # --- TESTING ---
        y_test_estimation = self.model.estimate(real_estate_test)
        metrics_test = self.model_metrics.compute_metrics(
             y_test_estimation, real_estate_test['price'],
        )
        logger.info(f"Computed metrics (test): {metrics_test}")

        logger.info("Saving model...")
        self.model_saver.save_model(
            model=self.model,
            result={
                'train': {'RMSE': metrics_train[0], 'MAPE': metrics_train[1], 'MAE': metrics_train[2]},
                'test': {'RMSE': metrics_test[0], 'MAPE': metrics_test[1], 'MAE': metrics_test[2]}
            },
        )
        logger.info("Model saved!")


def main() -> None:
    real_estate_model_creator = RealEstateModelCreator(
        loader = CSVLoader(),
        model  = RealEstateModel(),
        model_saver = ModelSaver(model_filename = './model/model.pkl', 
                                 result_filename = './model/result.pkl'),
        model_metrics = ModelMetrics())
    real_estate_model_creator.run()


if __name__ == "__main__":
    main()