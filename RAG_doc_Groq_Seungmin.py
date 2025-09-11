import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()

## os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_api_key=os.environ["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
groq_api_ke=os.environ["GROQ_API_KEY"]

llm=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_ke)

prompt=ChatPromptTemplate.from_template(
"""
Please find document link that the user searches and provide the link to the user.
Please ensure to answer as accurately as you can.

<context>
{context}
<context>

Question:{input}
"""
)

def create_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loadss=TextLoader("docs.txt")
        st.session_state.docs=st.session_state.loadss.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=10)
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(openai_api_key)


st.markdown("### <ins>Content Curation chat assistant</ins>", unsafe_allow_html=True)
doc_name=['Kuby','Hume Triage','Sports Schema','Sports Supplement','Common mistakes on Sports Recon jobs','Schema Notes for /lr_sports','Schema Notes for /olympics','Schema Notes for /tennis']
doc_name.insert(0, '')
user_prompt=st.selectbox(
    'Select the document (start typing to search):',
    doc_name)

## if st.button("Documents embeddings"):
    ## create_embeddings()
    ## st.write("The vector is ready.")


if user_prompt:
    create_embeddings()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response=retrieval_chain.invoke({'input':user_prompt})
    st.write("Here is the requested document:")
    st.write(response['answer'])

## else:
    ## st.write("Please enter a valid input.")