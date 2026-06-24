import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.set_page_config(page_title="Customer Service Chatbot")

st.title("🤖 Customer Service Chatbot")

if st.button("Create Knowledge Base"):
    create_vector_db()
    st.success("Knowledge Base Created Successfully!")

question = st.text_input("Ask your question")

if question:
    chain = get_qa_chain()
    response = chain.invoke({"query": question})

    st.subheader("Answer")
    st.write(response["result"])