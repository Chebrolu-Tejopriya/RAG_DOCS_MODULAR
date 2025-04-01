from langchain_huggingface import HuggingFaceEmbeddings

def embedding_function():
    """
    Initializes the embedding function using HuggingFaceEmbeddings.
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings