import streamlit as st
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from Utils.streamlit_data_handler import handle_uploaded_files


try:
    with st.sidebar:
        st.header("Upload Section")
        uploaded_files = st.file_uploader(key="streamlit_file_uploader",label="Upload files to application : ",type=['pdf','docx','txt'],accept_multiple_files=True)
        if uploaded_files:
            handle_uploaded_files(uploaded_files)
            st.success("Files uploaded")
except Exception as e:
    st.error(f"Exception occured : {e}")