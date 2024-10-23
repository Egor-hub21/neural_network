import neptune
import json
import os

name_file = 'neptune_run.json'
project="Egor-hub21/Test-project"
api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGI2MTc5OC01Njc3LTQ5NjAtYWU4OC1kNDVlYjVlZjU5OTAifQ=="

def check_neptune_run_file():
    """Проверяет наличие файла neptune_run.json.
    
    Returns:
        bool: True, если файл существует, иначе False.
    """
    return os.path.isfile(name_file)

def init_neptune():
    """Инициализирует новый run в Neptune.
    
    Returns:
        neptune.run.Run: Объект run.
    """
    run = neptune.init_run(
        project = project,
        api_token= api_token,
    )
    return run
    
def delete_neptune_run_file():
    """Удаляет файл neptune_run.json, если он существует."""
    if check_neptune_run_file():
        os.remove(name_file)
        
def save_neptune_run_file(run):
    """Сохранение идентификатора run в файл."""
    with open(name_file, 'w') as f:
        json.dump({'run_id': run._id,
                   'sys_id': run._sys_id}, f, indent=4)
        
def read_neptune_run():
    """Читает файл neptune_run.json и возвращает объект run.
    
    Returns:
        neptune.run.Run: Объект run, если файл существует, иначе None.
    """
    if check_neptune_run_file():
        with open(name_file, 'r') as f:
            sys_id = json.load(f)['sys_id']
        return neptune.init_run(
                    project = project,
                    api_token= api_token,
                    with_id=sys_id,
            )
    else:
        print("Файл neptune_run.json не найден. Не удалось получить run.")
        return None


    
    