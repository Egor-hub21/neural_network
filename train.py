import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import yaml
import json

import neptune
from neptune_tensorflow_keras import NeptuneCallback

run = neptune.init_run(
    project="Egor-hub21/Test-project",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGI2MTc5OC01Njc3LTQ5NjAtYWU4OC1kNDVlYjVlZjU5OTAifQ==",
)  # your credentials

def build_model() -> tf.keras.models.Model:
    
    # Создание модели
    model = keras.models.Sequential([
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1),
    ])
    
    
    # Компиляция модели (Выбор гиппер параметров)
    model.compile(
    optimizer='adam',
    loss='mse',
    metrics=["mse"]

)
    return model


def train(patch_data: str, patch_model: str,
          epochs: int, batch_size: int) -> None:
    
    # Names read files
    names = ['x_train','x_val','y_train','y_val']

    # Dictionary for storing data
    data = {}
    
    # Load data
    for name in names:
         data[name] = pd.read_csv(f"{patch_data}/{name}.csv")
    
    # Builder model
    model = build_model()
    
    # Создание callback 
    neptune_callback = NeptuneCallback(run=run)
    
    history = model.fit(
        data['x_train'],
        data['y_train'],
        validation_data = (data['x_val'], data['y_val']),
        epochs = epochs,
        batch_size = batch_size,
        shuffle =True,
        verbose = 0,
        callbacks = [neptune_callback],
        )
    
    metrics = {
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }
    
    with open('metrics.json', 'w') as file:
        json.dump(metrics, file, indent = 4)
    
    os.makedirs(patch_model, exist_ok=True)
    model.save(f'{patch_model}/model.keras')
    
    # Логирование модели    
    run['model'].upload(f'{patch_model}/model.keras')
    run.stop()

if __name__ == '__main__':
    
    with open ('params.yaml') as file:
        params = yaml.safe_load(file)
        
    with open ('paths.yaml') as file:
        paths = yaml.safe_load(file)
    
    # Логирование параметров    
    run['parameters'] = params['train']
        
    patch_data = paths['train']['patch_data']
    patch_model = paths['train']['patch_model']
    
    epochs = params['train']['epochs']
    batch_size = params['train']['batch_size']
    
    train(patch_data, patch_model,
          epochs, batch_size)
