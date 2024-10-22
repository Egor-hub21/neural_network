import pandas as pd
from tensorflow.keras.models import load_model

import yaml


def evaluate(patch_model: str, patch_data: str) -> None:
    
    # Load the model
    model = load_model(patch_model)
    
    # Load the data 
    x_test = pd.read_csv(f"{patch_data}/x_test.csv")
    y_test = pd.read_csv(f"{patch_data}/y_test.csv")

    # Evaluate the model
    loss, metrics = model.evaluate(x_test, y_test)
    
    print(f'Loss: {loss}')
    print(f'Metrics: {metrics}')

if __name__ == '__main__':
    
    with open ('params.yaml') as f:
        params = yaml.safe_load(f)
    
    patch_model = params['evaluate']['patch_model']
    patch_data = params['evaluate']['patch_data']
    evaluate(patch_model, patch_data)