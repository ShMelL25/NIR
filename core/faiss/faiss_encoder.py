from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.model_config import check_cuda

class Model_embeddings:
    
    def __init__(self):
        self.load_embeddings()
    
    def load_embeddings(self):
        
        model_path = 'config\models\llama-2-7b-chat.Q4_0\llama-2-7b-chat.Q4_0.gguf'
        if check_cuda() == True:
            self.embeddings = LlamaCppEmbeddings(
                                model_path=model_path,
                                n_gpu_layers=-1
                                                 )
        else:
            self.embeddings = LlamaCppEmbeddings(model_path=model_path)
    
    def transform(self, path_file:str):
        
        test_dict = {'pdf':PyPDFLoader(path_file)}
        text_splitter = RecursiveCharacterTextSplitter()
        docs = test_dict[path_file.split('.')[1]].load_and_split(text_splitter)
        
        faiss_docs = FAISS.from_documents(docs, self.embeddings)
        
        return faiss_docs
        