# from flask import Flask
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os

# app = Flask(__name__)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        return None
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),  # Ensure parentheses
        memory=memory
    )
    return conversation_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.write(response)

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userInput(user_question)
    
    st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)
    
    # with st.sidebar:
    #     st.subheader("Your documents")
    #     pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    #     if st.button("Process"):
    #         with st.spinner("Processing"):
    #             # Ambil text pdf
    #             raw_text = get_pdf_text(pdf_docs)
                
    #             # Ambil Text Chunks
    #             text_chunks = get_text_chunks(raw_text)
                
    #             # Vector Store
    #             vectorstore = get_vectorstore(text_chunks)
    #             if vectorstore is None:
    #                 st.error("Failed to create vector store. Please check your OpenAI API key.")
    #                 return
                
    #             # Create Conversation Chain
    #             st.session_state.conversation = get_conversation_chain(vectorstore)
    with st.sidebar:
        st.subheader("Your documents")
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not api_key:
                st.error("Please enter your OpenAI API key.")
                return
            
            with st.spinner("Processing"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    

if __name__ == "__main__":
    main()