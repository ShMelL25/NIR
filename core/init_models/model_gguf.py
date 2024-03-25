from llama_cpp import Llama
from NIR.config.model_config import check_cuda

class Model_GGUF:
    
    def __init__(self, model_path):
        if check_cuda() == True:
            self.model = Llama(
                    model_path=model_path,
                    n_ctx=16000,  # Context length to use
                    n_threads=32,            # Number of CPU threads to use
                    n_gpu_layers=-1        # Number of model layers to offload to GPU
                )
        else:
            self.model = Llama(
                    model_path=model_path,
                    n_ctx=16000,  # Context length to use
                    n_threads=32  # Number of model layers to offload to GPU
                )
        
    def predict(self, text:str, generation_kwargs):
        return self.model(text, **generation_kwargs)