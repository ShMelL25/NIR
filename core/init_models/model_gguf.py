from llama_cpp import Llama
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from ..tokenizer.tokenizer import Tokenize_Model

class Model_GGUF:
    
    def __init__(self, model_path:str):
        
        self.model = Llama(
                    model_path=model_path,
                    n_ctx=16000,  # Context length to use
                    n_threads=32  # Number of model layers to offload to GPU
                )
        self.embedding_model = Tokenize_Model(model_path=model_path)
        self.pipeline = HuggingFacePipeline(pipeline=pipeline(
                    "text-generation", model=self.model, max_new_tokens=300
                    ))
        
    def predict(self, text:str):
        return self.pipeline.predict(text)
