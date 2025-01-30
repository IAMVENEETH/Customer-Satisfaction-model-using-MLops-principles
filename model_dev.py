from abc import ABC,abstractmethod
import logging
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for model development
    """
    @abstractmethod
    def train_model(self,X_train,y_train):
        pass

class LinearRgressionModel(Model):
    def train_model(self,X_train,y_train):
        '''
        Train the model
        args:
            X_train: Training data
            y_train: Target data
        return: 
            Trained model
        '''
        try:
            model = LinearRegression()
            model.fit(X_train,y_train)
            logging.info('Model training completed')
            return model
        except Exception as e:
            logging.error(f'Error training model: {e}')
            raise e
        