from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from NIR.config.model_config import check_cuda
import os


class Model_embeddings:
    
    def __init__(self, model_path):
        self.load_embeddings(model_path=model_path)
    
    def load_embeddings(self, model_path):
        
        #model_path = 'config\models\llama-2-7b-chat.Q4_0\llama-2-7b-chat.Q4_0.gguf'
        if check_cuda() == True:
            self.embeddings = LlamaCppEmbeddings(
                                model_path=model_path,
                                n_gpu_layers=-1
                                                 )
        else:
            self.embeddings = LlamaCppEmbeddings(model_path=model_path)
    
    def transform(self, path_file:str, academic_subject:str):
        
        test_dict = {
            'pdf':PyPDFLoader(path_file),
            'txt':TextLoader(path_file),
            'docx':Docx2txtLoader(path_file)
            }
        
        file_name = path_file.split('.')
        
        text_splitter = RecursiveCharacterTextSplitter()
        docs = test_dict[file_name[1]].load_and_split(text_splitter)
        
        faiss_docs = FAISS.from_documents(docs, self.embeddings)
        self.save_embeddings(path_file=path_file, faiss_docs=faiss_docs, academic_subject=academic_subject)
        
        return faiss_docs
    
    def save_embeddings(self, path_file:str, faiss_docs, academic_subject:str):
        
        name = path_file.split('/')[-1]
        
        self.add_derectory(academic_subject)
        
        if not os.path.isdir(f"NIR/config/books/{academic_subject}/{name}"):
            os.mkdir(f"NIR/config/books/{academic_subject}/{name}")
        
        faiss_docs.save_local(f'NIR/config/books/{academic_subject}/{name}')
        
    def add_derectory(self, academic_subject):
        
        if not os.path.isdir(f"NIR/config/books/{academic_subject}"):
            os.mkdir(f"NIR/config/books/{academic_subject}")
            
        