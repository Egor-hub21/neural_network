import pandas as pd
from tensorflow.keras.models import load_model

import neptune
from neptune_tensorflow_keras import NeptuneCallback


import yaml
import json

import log

def evaluate(patch_model: str, patch_data: str,
             run) -> None:
    
    # Load the model
    model = load_model(patch_model)
    
    # Load the data 
    x_test = pd.read_csv(f"{patch_data}/x_test.csv")
    y_test = pd.read_csv(f"{patch_data}/y_test.csv")

    # Evaluate the model
    eval_metrics = model.evaluate(x_test, y_test, verbose=0)
    
    #Логирование функции потерь в Neptune
    for j, metric in enumerate(eval_metrics):
        run["eval/{}".format(model.metrics_names[j])] = metric
    
    with open('test_metrics.json', 'w') as file:
        json.dump(eval_metrics, file, indent = 4)
    

if __name__ == '__main__':
    
    with open ('paths.yaml') as file:
        paths = yaml.safe_load(file)
    
    run = None
    
    if log.check_neptune_run_file():
        run = log.read_neptune_run()
        log.delete_neptune_run_file()
    else:
        run = log.init_neptune()
    
    patch_model = paths['evaluate']['patch_model']
    patch_data = paths['evaluate']['patch_data']
    evaluate(patch_model, patch_data, run)
    
    run.stop()
    