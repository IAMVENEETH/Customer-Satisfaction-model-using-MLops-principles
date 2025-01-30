import logging
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE,RMSE,R2
from zenml.client import Client
import mlflow


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model:RegressorMixin,
    x_test:pd.DataFrame,
    y_test:pd.Series,              
    ) -> Tuple[
        Annotated[float, 'Mean Squared Error'],
        Annotated[float, 'R2 Score'],
        Annotated[float, 'Root Mean Squared Error']
    ]:
    '''
    Evaluates the model.
    args:
        data: pd.DataFrame: The data to evaluate on.
    returns:
        None
    '''
    logging.info('Evaluating model...')
    # Evaluate the model
    y_pred = model.predict(x_test)

    mse_class = MSE()
    r2_class = R2()
    rmse_class = RMSE()

    mse = mse_class.evaluate_model(y_test, y_pred)
    mlflow.log_metric("MSE", mse)
    r2 = r2_class.evaluate_model(y_test, y_pred)
    mlflow.log_metric("R2", r2)
    rmse = rmse_class.evaluate_model(y_test, y_pred)
    mlflow.log_metric("RMSE", rmse)

    return mse,r2,rmse