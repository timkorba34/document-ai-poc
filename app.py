import streamlit as st
import fitz
from PIL import Image
import io
import pandas as pd

# -------------------------
# FUNCTIONS
# -------------------------

# -------------------------
# Analyze Page
# -------------------------
def analyze_page(image, page_number, page_text):

    text = page_text.strip()
    lower_text = text.lower()

    score = 0

    # First page is always a new document
    if page_number == 1:
        score += 100

    # Shorter first-page style headers
    if len(text) < 1500:
        score += 10

    # Has title-like formatting clues
    first_lines = text.splitlines()[:5]
    first_block = " ".join(first_lines).lower()

    if len(first_lines) > 0:
        score += 10

    # Common document-start patterns, not document-specific
    generic_start_terms = [
        "date",
        "name",
        "id",
        "number",
        "account",
        "case",
        "invoice",
        "application",
        "report",
        "statement",
        "form"
    ]

    for term in generic_start_terms:
        if term in first_block:
            score += 5

    # Continuation indicators reduce score
    continuation_terms = [
        "continued",
        "page 2",
        "page 3",
        "page 4",
        "continued on next page"
    ]

    for term in continuation_terms:
        if term in lower_text:
            score -= 30

    # Very little text usually needs review
    if len(text) < 50:
        confidence = 60
    else:
        confidence = min(max(score, 0), 100)

    is_new_document = confidence >= 70

    return {
        "page": page_number,
        "is_new_document": is_new_document,
        "document_type": "Unknown",
        "confidence": confidence,
        "review_needed": confidence < 90,
        "text_preview": page_text[:150]
    }
# -------------------------
# Group Documents
# -------------------------

def group_documents(results):
    documents = []
    current_document = None

    for result in results:
        if result["is_new_document"] or current_document is None:
            current_document = {
                "document_number": len(documents) + 1,
                "document_type": result["document_type"],
                "start_page": result["page"],
                "end_page": result["page"],
                "confidence": result["confidence"],
                "review_needed": result["review_needed"]
            }
            documents.append(current_document)
        else:
            current_document["end_page"] = result["page"]

            if result["confidence"] < current_document["confidence"]:
                current_document["confidence"] = result["confidence"]

            if result["review_needed"]:
                current_document["review_needed"] = True

    return documents

# -------------------------
# Extract Text From Page
# -------------------------
def extract_text_from_page(page):
    text = page.get_text("text")
    return text.strip()

# -------------------------
# STREAMLIT UI
# -------------------------

st.set_page_config(
    page_title="Document AI PoC",
    layout="wide"
)

st.title("AI Document Segmentation PoC")

uploaded_file = st.file_uploader(
    "Upload a scanned PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf_bytes = uploaded_file.read()

    pdf_document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    total_pages = len(pdf_document)

    st.success(
        f"PDF loaded successfully ({total_pages} pages)"
    )

    st.subheader("Page Previews")

    results = []

    st.subheader("Grouped Documents")

    documents = group_documents(results)
    
    doc_df = pd.DataFrame(documents)
    
    st.dataframe(doc_df, use_container_width=True)

    for page_num in range(total_pages):

        page = pdf_document[page_num]
    
        pix = page.get_pixmap(
            matrix=fitz.Matrix(1.5, 1.5)
        )
    
        img_bytes = pix.tobytes("png")
    
        image = Image.open(
            io.BytesIO(img_bytes)
        )
    
        # OCR TEXT EXTRACTION
        page_text = extract_text_from_page(page)
    
        # AI ANALYSIS
        result = analyze_page(
            image,
            page_num + 1,
            page_text
        )
    
        results.append(result)
    
        st.image(
            image,
            caption=f"Page {page_num + 1}",
            width=350
        )
    
        st.write(result)
