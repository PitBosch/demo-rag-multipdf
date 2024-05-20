import streamlit as st
import os
from langchain import hub
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore

# Set API keys
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

# Initialize Pinecone
pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "question-answering"

# Vector store setup
vectorstore = PineconeVectorStore(index_name=index_name, embedding=OpenAIEmbeddings())

st.title("Question Answering App with LangChain")

# Upload PDF files
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits.extend(text_splitter.split_documents(docs))

    # Add to vector store and create retriever
    vectorstore.upsert_documents(all_splits)
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

    # Input question
    query = st.text_input("Enter your question:")
    if query:
        answer = rag_chain.invoke(query)
        st.write("Answer:", answer)
