import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Import a custom chunking strategy
from chunks import chunk_text
# Import the custom embedding function
from embedding import embedding_function

with open(r"Documents\pydantic_processed2.txt", "r", encoding="utf-8") as f:
    content = f.read()

#  Split the documentation with respect to the each web page

content = content.split("Page ")
content_pages = []
for page in content[2:]:
    content_pages.append(i[9:])


total_text_chunks = []
total_code_chunks = []

for page in content_pages:
    chunks = chunk_text(page, chunk_size = 1500)
    for i in chunks[0]:
        total_text_chunks.append(i)
    for i in chunks[1]:
        total_code_chunks.append(i)
    
print("Total text chunks: ", len(total_text_chunks))
print("Total code chunks: ", len(total_code_chunks))

# Initialize the embeddings function

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = embedding_function()

current_dir = os.path.dirname(os.path.abspath("__file__"))
db_dir = os.path.join(current_dir, "db")
persistant_directory = os.path.join(db_dir, "chroma_db")

if os.path.exists(persistant_directory):
    print("Chroma DB already exists")
else:
    print("Creating Chroma DB vector store")
    db  = Chroma.from_documents(
        total_text_chunks,
        embeddings,
        persistant_directory=persistant_directory,
    )
    print("Chroma DB vector store created")
    
