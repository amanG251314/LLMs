import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
load_dotenv()


embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_pdf_text(pdf_docs):
    text=''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size = 800, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embedding = HuggingFaceEmbeddings(model_name = embedding_model_name)
    #Create Vector Store
    vectorStore = FAISS.from_texts(text_chunks, embedding = embedding)
    return vectorStore

def get_conversation_chain(vectorstore):
    llm =HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever = vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2==0:
            #st.write(message)
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            #st.write(message)
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config("Chat with Multiple PDFs",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with Multiple PDFS :books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    user_question = st.text_input('Ask a question from your documents')
    if user_question:
        handle_user_input(user_question)
    with st.sidebar:
        st.header("Chat with PDF")
        st.title("LLM ChatApp using LangChain")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload the PDF Files here and Click on Process", accept_multiple_files=True)

        st.markdown(
            '''
            - [Stremlit](https://streamlit.io/)
            - [LangChain](https://www.langchain.com/)
            ''')
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract Text from PDF
                raw_text = get_pdf_text(pdf_docs=pdf_docs)

                # Split the Text into Chunks
                text_chunks = get_text_chunks(raw_text)
                #Create Vector Store
                vectorStore = get_vector_store(text_chunks=text_chunks)
                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vectorStore)
                st.success("Done!")


                

if __name__ == "__main__":
    main()