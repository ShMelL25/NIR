from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import LlamaCppEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from NIR.config.model_config import check_cuda
import os


class Model_embeddings:
    
    def __init__(self, model_path):
        self.load_embeddings(model_path=model_path)
    
    def load_embeddings(self, model_path):
    
        if check_cuda() == True:
            self.embeddings = LlamaCppEmbeddings(
                                model_path=model_path,
                                n_gpu_layers=-1
                                                 )
        else:
            self.embeddings = LlamaCppEmbeddings(model_path=model_path)
    
    def transform(self, path_file:str, chunk_size:int=100, chunk_overlap:int=20):
        
        test_dict = {
            'pdf':PyPDFLoader(path_file),
            'txt':TextLoader(path_file),
            'docx':Docx2txtLoader(path_file)
            }
        
        file_name = path_file.split('.')
        
        text_splitter = SemanticChunker(self.embeddings)
        docs = test_dict[file_name[1]].load_and_split(text_splitter)
        #faiss_docs = FAISS.from_documents(docs, self.embeddings)
        
        return docs
    
    def local_load(self, path_file:str):
          
        faiss_docs = FAISS.load_local(path_file, self.embeddings, allow_dangerous_deserialization=True)
        
        return faiss_docs
    
    def search_embeddings(self, query:str, faiss_docs:str):
        return faiss_docs.similarity_search(query)[0].page_content
        
    
    def save_embeddings(self, path_file:str, faiss_docs, academic_subject:str):
        
        name = path_file.split('/')[-1]
        
        self.add_derectory(academic_subject)
        
        if not os.path.isdir(f"NIR/config/books/{academic_subject}/{name.split('.')[0]}"):
            os.mkdir(f"NIR/config/books/{academic_subject}/{name.split('.')[0]}")
        path_save = f"NIR/config/books/{academic_subject}/{name.split('.')[0]}"
        
        faiss_docs.save_local(path_save)
        
    def add_derectory(self, academic_subject):
        
        if not os.path.isdir(f"NIR/config/books/{academic_subject}"):
            os.mkdir(f"NIR/config/books/{academic_subject}")
        
    
            
        