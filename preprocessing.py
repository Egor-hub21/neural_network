import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessing(path_data: str, patch_output: str, file_name_train:str,
                  file_name_val:str, file_name_test:str) -> None:
    """_summary_

    Args:
        path_data (str): _description_
        patch_output (str): _description_
        file_name_train (str): _description_
        file_name_val (str): _description_
        file_name_test (str): _description_
    """
    
    # создание директории для сохранения файла
    os.makedirs(patch_output, exist_ok=True)
    
    # Словарь хранит название файла и dataFrame
    data_dict = {}
    # Перебор всех файлов в указанной директории
    for filename in os.listdir(path_data):
    # Проверка, является ли файл CSV
        if filename.endswith('.csv'):
            # Полный путь к файлу
            file_path = os.path.join(path_data, filename)
            # Загрузка данных в DataFrame
            df = pd.read_csv(file_path)
            # Извлечение имени файла без расширения
            key = os.path.splitext(filename)[0]
            # Добавление в словарь
            data_dict[key] = df

    scaler = StandardScaler()
    
    scaler.fit(data_dict[file_name_train])
    
    features = [file_name_train, file_name_val, file_name_test]
      
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
        data_dataFrame[name].to_csv(f"{patch_output}/{name}.csv", index = False)    
 
 
    
if __name__ == '__main__':
    path_data = 'data/split' 
    patch_output = 'data/preprocessed'
    file_name_train = 'x_train'
    file_name_val = 'x_val'
    file_name_test = 'x_test'
    preprocessing(path_data, patch_output, file_name_train,
                  file_name_val, file_name_test)