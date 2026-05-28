import streamlit as st

st.set_page_config(page_title="Document AI PoC")

st.title("AI Document Segmentation PoC")

uploaded_file = st.file_uploader(
    "Upload a scanned PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success("PDF uploaded successfully")
    
    st.write("Filename:", uploaded_file.name)
    st.write("File Size:", uploaded_file.size, "bytes")
