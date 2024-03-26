from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from langchain_experimental.text_splitter import SemanticChunker

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

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique
    
    def embedding_documents(self, docs_processed):
        KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                docs_processed, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
        return KNOWLEDGE_VECTOR_DATABASE