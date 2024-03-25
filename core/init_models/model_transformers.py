from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
import torch

class Model:
    
    def __init__(self, model_path, low_cpu_mem_usage:bool=False, 
                 return_dict:bool=False, torch_dtype=torch.float32,
                 load_in_4bit:bool=False, load_in_8bit:bool=False, 
                 device_map:str='auto'):
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                           low_cpu_mem_usage=low_cpu_mem_usage,
                                                            return_dict=return_dict,
                                                            torch_dtype=torch_dtype,
                                                            load_in_4bit=load_in_4bit,
                                                            load_in_8bit=load_in_8bit,
                                                            device_map=device_map
                                                            )
        pipe = pipeline(
                    "text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=300
                    )
        self.pipeline = HuggingFacePipeline(pipeline=pipe)
        
    def predict(self, text:str):
        return self.pipeline.predict(text)