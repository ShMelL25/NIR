from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import torch

class Model:
    
    def __init__(self, model_path, low_cpu_mem_usage:bool=False, 
                 return_dict:bool=False, torch_dtype=torch.float32,
                 load_in_4bit:bool=False, load_in_8bit:bool=False, 
                 device_map:str='auto'):
        
        self.embedding_model = HuggingFaceEmbeddings(
                                                    model_name=model_path,
                                                    multi_process=True,
                                                    model_kwargs={"device": "cuda"},
                                                    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)
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
    
    def split_documents(self,chunk_size,knowledge_base):
            
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique
    
    def embedding_documents(self, docs_processed, ):
        KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                docs_processed, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
        return KNOWLEDGE_VECTOR_DATABASE