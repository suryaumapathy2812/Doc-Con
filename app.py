# Reference
# https://www.youtube.com/watch?v=dXxQ0LR-3Hg

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_chat import message
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Instantiate the model. Callbacks support token-wise streaming
# model = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=512, n_threads=8)

# Generate text
# response = model("Once upon a time, ")


def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunk(raw_texts):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunk = text_splitter.split_text(raw_texts)
    return chunk


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    # llm = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=512, n_threads=8)
    # llm = ChatOpenAI(model="gpt-4")
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    # incase to log the responses
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]

    for i, chat in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # User questions
            message(chat.content, is_user=True)
        else:
            # Bot response
            message(chat.content, is_user=False)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with PDF :books:")
    user_question = st.text_input("Ask any question about Java 8")

    if user_question:
        handle_user_question(user_question)

    # message("Hello There, How can I assist you")

    with st.sidebar:
        st.header("Your document")
        pdf_docs = st.file_uploader("Upload you pdf file", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get text chunks
                text_chunk = get_text_chunk(raw_texts=raw_text)
                st.write(text_chunk)

                # create vector store
                vector_store = get_vector_store(text_chunk)

                # create a conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()
