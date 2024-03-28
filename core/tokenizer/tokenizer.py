from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
import os

class Tokenize_Model:
    
    def __init__(self, model_path:str):
        
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                                        model_name=model_path,
                                        multi_process=True,
                                        model_kwargs={"device": "cuda"},
                                        encode_kwargs={"normalize_embeddings": True})
        except OSError:
            self.embedding_model = LlamaCppEmbeddings(model_path=model_path)
        
        self.text_splitter = SemanticChunker(self.embedding_model)
            
    def split_documents(self, path_file:str):
        
        test_dict = {
            'pdf':PyPDFLoader(path_file),
            'txt':TextLoader(path_file),
            'docx':Docx2txtLoader(path_file)
            }
        file_name = path_file.split('.')
        docs = test_dict[file_name[1]].load_and_split(self.text_splitter)
        
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique
    
    def embedding_documents(self, docs_processed):
        knowledge_vector_database = FAISS.from_documents(
                docs_processed, 
                self.embedding_model, 
                distance_strategy=DistanceStrategy.COSINE
            )
        return knowledge_vector_database

    def save_embeddings(self, faiss_docs, academic_subject:str):
        
        if not os.path.isdir(f'../../config/book/{academic_subject}'):
            os.mkdir(f'../../config/book/{academic_subject}')
            
        faiss_docs.save_local(f'../../config/book/{academic_subject}')
    
    def local_load(self, academic_subject:str):
        path_file = f'../../config/book/{academic_subject}'
        faiss_docs = FAISS.load_local(path_file, 
                                      self.embeddings, 
                                      allow_dangerous_deserialization=True
                                      )
        
        return faiss_docs
    
    def search_embeddings(self, query:str, faiss_docs):
        return faiss_docs.similarity_search(query)[0].page_content