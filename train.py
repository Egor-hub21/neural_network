import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import yaml

def build_model() -> tf.keras.models.Model:
    
    # Создание модели
    model = keras.models.Sequential([
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1),
    ])
    
    
    # Компиляция модели (Выбор гипер параметров)
    model.compile(
    optimizer='adam',
    loss='mse',
    metrics=["mse"]

)
    return model


def train(patch_data: str, patch_model: str) -> None:
    
    # Names read files
    names = ['x_train','x_val','y_train','y_val']

    # Dictionary for storing data
    data = {}
    
    # Load data
    for name in names:
         data[name] = pd.read_csv(f"{patch_data}/{name}.csv")
    
    # Builder model
    model = build_model()
    
    model.fit(
        data['x_train'],
        data['y_train'],
        validation_data = (data['x_val'], data['y_val']),
        epochs = 20,
        batch_size = 32,
        shuffle =True,
        verbose = 1,
        )
    
    os.makedirs(patch_model, exist_ok=True)
    model.save(f'{patch_model}/model.keras')

if __name__ == '__main__':
    
    with open ('params.yaml') as f:
        params = yaml.safe_load(f)
    
    patch_data = ['train']['patch_data']
    patch_model = ['train']['patch_model']
    train(patch_data, patch_model)
