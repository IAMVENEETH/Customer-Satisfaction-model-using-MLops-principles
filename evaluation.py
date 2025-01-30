from abc import ABC,abstractmethod
from sklearn.metrics import r2_score,root_mean_squared_error
import logging
import numpy as np

class Evaluation(ABC):
    """
    Abstract class for model evaluation
    """
    @abstractmethod
    def evaluate_model(self,y_true:np.ndarray,y_pred:np.ndarray): 
        '''
        Evaluate the model
        args:
            y_true: Actual target data
            y_pred: Predicted target data
        return:
            Evaluation metrics
        '''
        pass

class MSE(Evaluation):
    '''
    Evaluation Strategy for Mean Squared Error
    '''
    def evaluate_model(self, y_true, y_pred):
        '''
        Evaluate the model
        args:
            y_true: Actual target data
            y_pred: Predicted target data
        return:
            Evaluation metrics
        '''
        try:
            logging.info('Evaluating model using Mean Squared Error')
            mse = root_mean_squared_error(y_true, y_pred)**2
            logging.info(f'Mean Squared Error: {mse}')
            return mse
        except Exception as e:
            logging.error(f'Error occured while evaluating model using Mean Squared Error: {str(e)}')
            raise e

class R2(Evaluation):
    '''
    Evaluation Strategy for R2 Score
    '''
    def evaluate_model(self, y_true, y_pred):
        '''
        Evaluate the model
        args:
            y_true: Actual target data
            y_pred: Predicted target data
        return:
            Evaluation metrics
        '''
        try:
            logging.info('Evaluating model using R2 Score')
            r2 = r2_score(y_true, y_pred)
            logging.info(f'R2 Score: {r2}')
            return r2
        except Exception as e:
            logging.error(f'Error occured while evaluating model using R2 Score: {str(e)}')
            raise e

class RMSE(Evaluation):
    '''
    Evaluation Strategy for Root Mean Squared Error
    '''
    def evaluate_model(self, y_true, y_pred):
        '''
        Evaluate the model
        args:
            y_true: Actual target data
            y_pred: Predicted target data
        return:
            Evaluation metrics
        '''
        try:
            logging.info('Evaluating model using Root Mean Squared Error')
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f'Root Mean Squared Error: {rmse}')
            return rmse
        except Exception as e:
            logging.error(f'Error occured while evaluating model using Root Mean Squared Error: {str(e)}')
            raise e
