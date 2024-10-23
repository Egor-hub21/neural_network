import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from typing import Union

import yaml
import json

import neptune
from neptune_tensorflow_keras import NeptuneCallback

run = neptune.init_run(
    project="Egor-hub21/Test-project",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGI2MTc5OC01Njc3LTQ5NjAtYWU4OC1kNDVlYjVlZjU5OTAifQ==",
)  # your credentials

def build_model(params_deep_layers: dict[str, dict[str, Union[str, int]]],
    params_compile: dict[str, Union[str, list[str]]]) -> tf.keras.models.Model:
    
    # Создание модели
    model = Sequential()
    
    for layer in params_deep_layers.values():
        model.add(Dense(layer['count_neuron'],
                        activation = layer['activation']))
    
    model.add(Dense(1))
    
    # Компиляция модели (Выбор hyper params)
    model.compile(
    optimizer = params_compile['optimizer'],
    loss = params_compile['loss'],
    metrics = params_compile['metrics'],

)
    return model


def train(patch_data: str, patch_model: str,
          params_fit: dict[str, Union[bool, int, str]],
          params_deep_layers: dict[str, dict[str, Union[str, int]]],
          params_compile: dict[str, Union[str, list[str]]]) -> None:
    
    # Names read files
    names = ['x_train','x_val','y_train','y_val']

    # Dictionary for storing data
    data = {}
    
    # Load data
    for name in names:
         data[name] = pd.read_csv(f"{patch_data}/{name}.csv")
    
    # Builder model
    model = build_model(params_deep_layers, params_compile)
    
    # Создание callback 
    neptune_callback = NeptuneCallback(run=run)
    
    history = model.fit(
        data['x_train'],
        data['y_train'],
        validation_data = (data['x_val'], data['y_val']),
        epochs = params_fit['epochs'],
        batch_size = params_fit['batch_size'],
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
    
    params_deep_layers = params['train']['deep_layers']
    params_compile = params['train']['compile']
    params_fit = params['train']['fit']
    
    train(patch_data, patch_model, params_fit, 
          params_deep_layers, params_compile)
