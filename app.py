import streamlit as st
import requests

# Title
st.set_page_config(page_title="📚 DSA RAG ChatBot", page_icon="✍🏼")
st.title("💻📚✍🏼 DSA RAG ChatBot")

# Input box
user_input = st.text_input("💬 Write a Topic Name...")

if user_input:
    with st.spinner("⌛ Generating Your Answer... Please Wait"):
        API_URL = "http://127.0.0.1:8000/query"
        # Call FastAPI endpoint
        response = requests.post(
            API_URL,
            json={"text": user_input}
        )

    if response.status_code == 200:
        answer = response.json()["answer"]
        st.markdown(f"**Answer:** {answer}")
    else:
        st.error("Error: Unable to get response from API")
