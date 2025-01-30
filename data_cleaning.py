import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class for data cleaning strategies.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessing(DataStrategy):
    """
    Data cleaning strategy for pre-processing.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Pre-processes the data.
        """
        try:
            data = data.drop({
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            },axis=1)

            data["product_weight_g"].fillna(data["product_weight_g"].mean(),inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].mean(),inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].mean(),inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].mean(),inplace=True)
            data["review_comment_message"].fillna("No Comment",inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop=["order_item_id","customer_zip_code_prefix"]
            data = data.drop(cols_to_drop,axis=1)
            return data
        except Exception as e:
            logging.error(f'Error pre-processing data: {e}')
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Data cleaning strategy for dividing the data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Divides the data into training and testing sets.
        """
        try:
            X = data.drop("review_score",axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Error dividing data: {e}')
            raise e
        
class DataCleaning:
    """
    Class to clean the data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """
        Cleans the data.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f'Error cleaning data: {e}')
            raise e
    

