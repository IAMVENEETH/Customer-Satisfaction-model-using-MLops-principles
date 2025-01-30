from zenml.steps.base_step import BaseStep

class ModelParameter(BaseStep):

    def __init__(self, model_name: str = "LinearRegression"):
        self.model_name = model_name