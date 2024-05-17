import streamlit as st
import os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup Streamlit
st.title("Question Answering App")

# Set API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load and display data
st.write("Loading blog content...")
loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",))
docs = loader.load()

# Split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Store data
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Input query
query = st.text_input("Enter your question:")
if query:
    # Retrieve and generate answer
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(query)
    
    # Display answer
    st.write("Answer:", answer)
