from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    df = ingest_df(datapath=data_path)
    X_train, X_test, y_train, y_test = clean_df(data=df)
    config = {"model_name":"LinearRegression"}
    model = train_model(X_train,X_test,y_train,y_test,config)
    mse, r2, rmse = evaluate_model(model, X_test, y_test)