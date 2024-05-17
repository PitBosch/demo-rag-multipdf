import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
import sys

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["OPENAI_API_KEY"]=st.secrets["openai_api_key"]
# Streamlit setup
st.title("Question Answering App with LangChain")
__import__('pysqlite3')

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Upload PDF files

from langchain_openai import ChatOpenAI



uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_splits = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load document
        loader = PyMuPDFLoader(uploaded_file.name)
        data = loader.load()

        # Split document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = text_splitter.split_documents(data)
        all_splits.extend(splits)

    # Add to vector store
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    # RAG prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    model = ChatOpenAI(model="gpt-3.5-turbo-0125")

    # RAG chain
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    # Add typing for input
    class Question(BaseModel):
        __root__: str

    chain = chain.with_types(input_type=Question)

    # Input question
    query = st.text_input("Enter your question:")
    if query:
        answer = chain.invoke({"__root__": query})
        st.write("Answer:", answer)
