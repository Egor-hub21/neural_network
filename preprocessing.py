import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

import yaml

def preprocessing(path_data: str, patch_output: str) -> None:
    """_summary_

    Args:
        path_data (str): _description_
        patch_output (str): _description_
    """
    
    # Создание директории для сохранения файла
    os.makedirs(patch_output, exist_ok=True)
    
    features = ['x_train','x_val','x_test']
    targets = ['y_train','y_val','y_test']
    
    # Хранит название файла и dataFrame
    data_dict = {}
    
    # Read files
    for file_name in features + targets:
        file_path = os.path.join(path_data, f'{file_name}.csv')
        if os.path.isfile(file_path):
            data_dict[file_name] = pd.read_csv(file_path)
        else:
            print(f'!!!_Файл {file_path} не найден_!!!')

    scaler = StandardScaler()
    
    # Training StandardScaler 
    scaler.fit(data_dict[features[0]])
      
    # 
    data_dataFrame = {}
    
    for name, data in data_dict.items():
        
        # Transform the training, test, validation data.
        if name in features:
           data_scaled  = scaler.transform(data)
           # Convert to dataFrame
           data_dataFrame[name] = pd.DataFrame(data_scaled,
                                               columns = data.columns)
        else:
            data_dataFrame[name] = data                 
                
        # Save
        data_dataFrame[name].to_csv(f"{patch_output}/{name}.csv",
                                    index = False)    
 
 
    
if __name__ == '__main__':
    
    with open ('paths.yaml') as file:
        paths = yaml.safe_load(file)
        
    path_data = paths['preprocessing']['path_data']
    patch_output = paths['preprocessing']['patch_output']

    preprocessing(path_data, patch_output)