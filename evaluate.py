import pandas as pd
from tensorflow.keras.models import load_model

import yaml
import json


def evaluate(patch_model: str, patch_data: str) -> None:
    
    # Load the model
    model = load_model(patch_model)
    
    # Load the data 
    x_test = pd.read_csv(f"{patch_data}/x_test.csv")
    y_test = pd.read_csv(f"{patch_data}/y_test.csv")

    # Evaluate the model
    loss, metric = model.evaluate(x_test, y_test)
    
    metrics = {
        'Metric': metric,
        'loss': loss
    }
    
    with open('test_metrics.json', 'w') as file:
        json.dump(metrics, file, indent = 4)
    

if __name__ == '__main__':
    
    with open ('paths.yaml') as file:
        paths = yaml.safe_load(file)
    
    patch_model = paths['evaluate']['patch_model']
    patch_data = paths['evaluate']['patch_data']
    evaluate(patch_model, patch_data)