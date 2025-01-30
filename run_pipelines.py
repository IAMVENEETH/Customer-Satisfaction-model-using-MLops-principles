import logging
logging.basicConfig(level=logging.DEBUG)
from zenml.client import Client
from pipelines.train_pipeline import training_pipeline


if __name__ == '__main__':


    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="/Users/veneeth/Desktop/PROJECT/data/olist_customers_dataset.csv")