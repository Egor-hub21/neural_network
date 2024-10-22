import pandas as pd
from tensorflow.keras.models import load_model


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
    patch_model = 'models/model.keras'
    patch_data = 'data/preprocessing' 
    evaluate(patch_model, patch_data)