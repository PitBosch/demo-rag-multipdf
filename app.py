import streamlit as st

import os 
import sys
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings


os.environ["OPENAI_API_KEY"]=st.secrets["openai_api_key"]
# Streamlit setup

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb


# Upload PDF files

from langchain_openai import ChatOpenAI


st.title("Question Answering App with LangChain")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_splits = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load document
        loader = PyMuPDFLoader(uploaded_file.name)
        docs = loader.load()

        # Split document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

    # Add to vector store
    vectorstore = Chroma.from_documents(
        documents=all_splits,
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
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # Add typing for input
    #class Question(BaseModel):
    #    __root__: str

    #chain = chain.with_types(input_type=Question)

    # Input question
    query = st.text_input("Enter your question:")
    if query:
        answer = rag_chain.invoke({"__root__": query})
        st.write("Answer:", answer)
