from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from embedding import embedding_function
from typing import List
from langchain_core.documents import Document

def retrive_chunks(query: str, top_k: int = 3, threshold: float = 0.2) -> List[Document]:
    """
    Retrieves the top K chunks from the Chroma DB based on the query.
    """
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    db_dir = os.path.join(current_dir, "db")
    persistant_directory = os.path.join(db_dir, "chroma_db")
    embeddings = embedding_function()

    if os.path.exists(persistant_directory):
        print("Vector Store Exists Retrieving chunks")
        db = Chroma(persistant_directory=persistant_directory, embedding_function=embeddings)

        retriver = db.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": threshold}
        )

        results = retriver.invoke(query)

    else: 
        print("Chroma DB vector store does not exist")
        return []