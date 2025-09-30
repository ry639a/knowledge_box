import streamlit as st
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import pypdf
import os

def generate_response(uploaded_file, question, open_api_key):
    if uploaded_file is not None:
        reader = pypdf.PdfReader(uploaded_file)
        print(reader.pages)
        documents = [Document(page.extract_text()) for page in reader.pages]
            #[uploaded_file.read().decode("utf-8")]
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        llm = ChatOpenAI(model="gpt-4o")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        database = Chroma.from_documents(texts, embeddings)
        retriever = database.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(question)
        return response

st.set_page_config(page_title="Ravali's Knowledge Box")
st.title("Knowledge Box!")
st.write("Welcome to Knowledge Box!")
st.write("Upload personal documents and get intelligent answers to questions from the documents!")

# run the script to view app on browser use below command.
# streamlit run D:\Interview_prep\prep\python\RAG\front_end.py

image_filename = "knowledge_box.jpg"
st.image(image_filename, use_container_width=False, width=300)

uploaded_file = st.file_uploader("Upload your file", type=["csv", "txt", "pdf"])
if uploaded_file is not None:
    st.write("File is uploaded successfully!")

question = st.text_input("Enter your question here", value="")
result = None
with st.form(key='qa_form', clear_on_submit=False, border=False):
    open_api_key = st.text_input("Enter your API key", type='password', disabled=not (uploaded_file and question))
    os.environ['OPENAI_API_KEY'] = open_api_key
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and question))
    if submitted and open_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, question, open_api_key)
            result = response
if result:
    st.info(result)