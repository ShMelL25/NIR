from tokenizer.tokenizer import Tokenize_Model

class Documents:
    
    def __init__(self, model_path):
        self.model = Tokenize_Model(model_path=model_path)
    
    def add_documents(self, path_file:str, academic_subject:str):
        docs = self.model.split_documents(path_file=path_file)
        faiss_db = self.model.embedding_documents(docs_processed=docs)
        self.model.save_embeddings(faiss_docs=faiss_db, 
                                   academic_subject=academic_subject)
        
    def search_text(self, academic_subject:str, query:str):
        faiss_db = self.model.local_load(academic_subject=academic_subject)
        answer = self.model.search_embeddings(query=query, faiss_docs=faiss_db)
        return answer