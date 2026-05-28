import streamlit as st
import fitz
from PIL import Image
import io

st.set_page_config(page_title="Document AI PoC", layout="wide")

st.title("AI Document Segmentation PoC")

uploaded_file = st.file_uploader("Upload a scanned PDF", type=["pdf"])

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)

    st.success(f"PDF loaded successfully ({total_pages} pages)")

    st.subheader("Page Previews")

    for page_num in range(total_pages):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        st.image(image, caption=f"Page {page_num + 1}", width=350)
