import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
import PyPDF2
import os
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
# This block of code imports pysqlite3 and sets it to sqlite3 to avoid potential conflicts
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Function to extract text from PDF files and return as a list
def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]
            text = pageObj.extract_text()
            text_list.append(text)
            sources_list.append(file.name + "_page_" + str(i))
    return [text_list, sources_list]

st.set_page_config(layout="centered", page_title="GoldDigger")
st.header("GoldDigger")
st.write("---")

# File uploader
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf"])
st.write("---")

if not uploaded_files:
    st.info("Upload files to analyze")
else:
    st.write(f"{len(uploaded_files)} document(s) loaded..")
    textify_output = read_and_textify(uploaded_files)
    
    documents = textify_output[0]
    sources = textify_output[1]
    
    # Extract embeddings
    key = os.environ['OPENAI_API_KEY']
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    
    # Vector store with metadata (page numbers)
    vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
    
    # Deciding model
    model_name = "gpt-3.5-turbo"
    # model_name = "gpt-4"
    
    retriever = vStore.as_retriever()
    retriever.search_kwargs = {'k': 2}
    
    # Initiate model
    llm = OpenAI(model_name=model_name, openai_api_key=key, streaming=True)
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")
    
    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = model({"question": user_q}, return_only_outputs=True)
                st.subheader('Your response:')
                st.write(result['answer'])
                st.subheader('Source pages:')
                st.write(result['sources'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Oops, the GPT response resulted in an error :( Please try again with a different question.")
