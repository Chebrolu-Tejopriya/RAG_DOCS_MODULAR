import time
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from retrieve_chunks import retrive_chunks
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

MongoClient = MongoClient("mongodb+srv://yogesh:12346789@cluster0.mongodb.net/test?retryWrites=true&w=majority")

# Initialize mongo cloud
db = MongoClient["code"]
collection = db["chunks"]



# Let's build streamlit UI
st.title("🔎 AI-Powered Documentation Assistant")

query = st.text_input("Enter your query: ")

if query:
    with st.spinner("Retrieving relevant chunks..."):
        retrieved_chunks = retrive_chunks(query, top_k=2, threshold=0.2)
        if not retrieved_chunks:
            st.warning("No relevant chunks found.")
        else:
            st.success(f"Found {len(retrieved_chunks)} relevant chunks.")
            st.subheader("Relevant Text Chunks Retrieved...")
            text = ""
            for i, doc in enumerate(retrieved_chunks):
                st.markdown(f"**Text Chunk-{i+1}:** {doc.page_content}")
                text += doc.page_content + "\n\n"
    
    with st.spinner("Attaching code chunks..."):
        # Attach code chunks to the retrieved text chunks
        if retrieved_chunks:
            parent_text_id = retrieved_chunks[0].metadata["chunk_id"]
            code_chunks = collection.find({"metadata.parent_text_id": parent_text_id})
            code_chunks = [code_chunk["page_content"] for code_chunk in code_chunks[:4]]
            st.subheader("Code Chunks Retrieved...")
            code_content = ""
            for i, code_chunk in enumerate(code_chunks):
                st.markdown(f"**Code Chunk-{i+1}:** {code_chunk}")
                code_content += code_chunk + "\n\n"
            st.subheader("Code Chunks Attached to the Text Chunks...")
    
    with st.spinner("Creating Prompt..."):
        # Create a prompt for the model
        context = (
            "Here are some documents that might help answer the question:"
            + query
            + "\n\nRelevant Text from pydantic documentation:\n"
            + text
            + "\n\nRelevant code"
            + code_content
            + "The code might contain syntax errors so correct it as needed"    
        )

    with st.spinner("Generating response..."):
        # Use Google Generative AI to generate a response
        model = ChatGoogleGenerativeAI(
            model = "gemini-1.5-pro",
            temeprature = 0.2,
        )

        messages = [
            SystemMessage(content="You are python expert and you can we understand errors in code and able to provide right syntax. Your task is to go through code and text from documentation you are provided with as content and able to answer user's question about code documentation. Help him explore documentation easily and explain the code you are provided with"),
            HumanMessage(content=context)
        ]

        result = model.invoke(messages)
        print(result.content)
        st.markdown(result.content)
        st.success("Response generated successfully!")
        


        


