import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(1):
    url = st.sidebar.text_input(f"URL {i}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(model_name = "gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=500)

if process_url_clicked:
    try:
        # load data
        loader = SeleniumURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            #separators=['\n\n', '\n', '.', ','],
            #is_separator_regex=False,
            separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200B",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
    ],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        # create embeddings and save it to FAISS index
        embeddings = HuggingFaceEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
    except Exception as e:
        st.error(f"Error processing URLs: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            try:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                # all_text = "\n".join([doc.page_content for doc in docs]) 
                # result = chain({"question": query, "text": all_text}, return_only_outputs=True)  
                # result will be a dictionary of this format --> {"answer": "", "sources": [] }
                st.header("Answer")
                st.write(result["answer"])
                

            #     # Display sources, if available
            #     sources = result.get("sources", "")
            #     if sources:
            #         st.subheader("Sources:")
            #         sources_list = sources.split("\n")  # Split the sources by newline
            #         for source in sources_list:
            #             st.write(source)
            except Exception as e:
                st.error(f"Error retrieving answer: {e}")
