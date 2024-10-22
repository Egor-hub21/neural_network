import os
import pandas as pd 
from sklearn.model_selection import train_test_split

import yaml

def split(path: str, path_to_save:str, test_size:float,
          val_size: float, name_target: str, number_random: int):
    """_summary_

    Args:
        path (str): _description_
        path_to_save (str): _description_
        test_size (float): _description_
        val_size (float): _description_
        name_target (str): _description_
        number_random (int): _description_
    """
    # создание директории для сохранения файла
    os.makedirs(path_to_save, exist_ok=True)
    
    # чтение файла
    data = pd.read_csv(path)
    
    # Разбиение на target и features
    target = data[[name_target]]
    features = data.drop(columns = [name_target])
    
    x_train, x_test_val, y_train, y_test_val = train_test_split(
        features, target, test_size = test_size + val_size,
        random_state = number_random
    )
    
    x_test, x_val, y_test, y_val = train_test_split(
        x_test_val, y_test_val,
        test_size = val_size / (test_size + val_size),
        random_state = number_random
    )
    
    split_data = {'x_train': x_train,
                  'x_val': x_val,
                  'x_test': x_test,
                  'y_train': y_train,
                  'y_val': y_val,
                  'y_test': y_test,} 
    
    # Сохранение данных
    for key, value in split_data.items():
        value.to_csv(f"{path_to_save}/{key}.csv", index = False) 
    
if __name__ == "__main__":
    
    with open ('params.yaml') as f:
        params = yaml.safe_load(f)
    
    path = params['split']['path']
    path_to_save = params['split']['path_to_save']
    test_size = params['split']['test_size']
    val_size = params['split']['val_size']
    name_target = params['split']['name_target']
    number_random = params['split']['number_random']
    split(path, path_to_save, test_size,
          val_size, name_target, number_random)