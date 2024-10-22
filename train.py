import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
    
    # Load data
    x_train = pd.read_csv(f"{patch_data}/x_train.csv")
    x_val = pd.read_csv(f"{patch_data}/x_val.csv")
    y_train = pd.read_csv(f"{patch_data}/y_train.csv")
    y_val = pd.read_csv(f"{patch_data}/x_val.csv")
    
    # Builder model
    model = build_model()
    
    model.fit(
        x_train,
        y_train,
        validation_data = (x_val, y_val),
        epochs = 20,
        batch_size = 32,
        shuffle =True,
        verbose = 1,
        )
    
    os.makedirs(patch_model, exist_ok=True)
    model.save(f'{patch_model}/model.keras')

if __name__ == '__main__':
    patch_data = 'data/preprocessed'
    patch_model = 'models'
    train(patch_data, patch_model)
