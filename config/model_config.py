import torch
import json

def check_cuda():
    return torch.cuda.is_available()

def open_json(path_file:str):
    
    with open(path_file) as json_data:
        data = json.loads(json_data)
        json_data.close()
        
    return data