import logging
from zenml import step
import pandas as pd
from src.data_cleaning import DataCleaning,DataPreProcessing,DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
    ]:

    try:
        process_stategy = DataPreProcessing()
        data_cleaning  =  DataCleaning(data,process_stategy)
        processed_data = data_cleaning.handle_data()
        
        divide_startegy = DataDivideStrategy()
        data_cleaning  =  DataCleaning(processed_data,divide_startegy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info('Data cleaning completed')
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(f'Error cleaning data: {e}')
        raise e