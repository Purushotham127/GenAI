import streamlit as st
import logging
import os
from pathlib import Path
import sys

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(
    lambda record: not (
        record.levelno == logging.WARNING
        and "missing ScriptRunContext! This warning can be ignored when running in bare mode."
        in record.getMessage()
    )
)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from Utils.streamlit_data_handler import handle_uploaded_files, get_llm_response, get_topk_results

try:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello. What would you like to know about your documents?"}
        ]

    with st.sidebar:
        st.header("Upload Section")
        uploaded_files = st.file_uploader(key="streamlit_file_uploader",label="Upload files to application : ",type=['pdf','docx','txt'],accept_multiple_files=True)
        if st.button("Upload files", icon=":material/upload:", disabled=not uploaded_files):
            with st.spinner("Ingesting data to Qdrant vectorDB..."):
                for uploaded_file in uploaded_files:
                    uploaded_file.seek(0)
                handle_uploaded_files(uploaded_files)
            st.success("Files uploaded")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("references"):
                with st.expander("Output references"):
                    st.write(message["references"])

    user_query = st.chat_input("Ask a question about your documents")
    if user_query and user_query.strip():
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            results_data = None
            try:
                with st.spinner("Generating response..."):
                    results = get_topk_results(user_query)
                    results_data, output = get_llm_response(user_query, results)
                    response_content = output.content
            except Exception as e:
                response_content = f"Unable to generate a response: {e}"
            st.markdown(response_content)
            # if results_data:
            #     with st.expander("Output references"):
            #         st.write(results_data)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": response_content,
                # "references": results_data
            })
except Exception as e:
    st.error(f"Exception occured : {e}")