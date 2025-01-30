import logging
from zenml import step
import pandas as pd
from src.model_dev import LinearRgressionModel
from steps.config import ModelParameter
from sklearn.base import RegressorMixin
from .config import ModelParameter
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train:pd.DataFrame,
    X_test:pd.DataFrame,
    y_train:pd.Series,
    y_test:pd.Series,
    config:dict
    ) -> RegressorMixin:
    '''
    Trains the model.
    args:
        data: pd.DataFrame: The data to train on.
    returns:
        None
    '''
    logging.info('Training model...')   
    # Train the model
    try:
        model=None
        if config["model_name"]=="LinearRegression":
            model = LinearRgressionModel()
            mlflow.sklearn.autolog()
            trained_model = model.train_model(X_train,y_train)
            return trained_model
        else:
            raise ValueError(f'Invalid model name: {config.model_name}')
    except Exception as e:
        logging.error(f'Error training model: {e}')
        raise