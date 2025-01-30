import logging
from zenml import step
import pandas as pd

class IngestData:
    '''
    Ingests the data from the source.
    '''
    def __init__(self,data_path:str):
        self.data_path = data_path

    def run(self) -> pd.DataFrame:
        '''
        Ingests the data from the source.
        '''
        logging.info('Ingesting data...')
        # Load the data
        data = pd.read_csv(self.data_path)
        return data

@step
def ingest_df(datapath:str) -> pd.DataFrame:
    '''
    Ingests the data from the source.
    args:
        datapath: str: The path to the data.
    returns:
        pd.DataFrame: The data.
    '''
    try:
        ingest_data_obj = IngestData(datapath)
        data = ingest_data_obj.run()
        return data
    except Exception as e:
        logging.error(f'Error ingesting data: {e}')
        raise e

