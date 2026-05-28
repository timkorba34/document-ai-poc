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
def analyze_page(image, page_number, page_text, previous_page_text=""):

    text = page_text.strip()
    previous_text = previous_page_text.strip()

    lower_text = text.lower()
    lower_previous = previous_text.lower()

    score = 0
    reasons = []

    if page_number == 1:
        score = 100
        reasons.append("First page in batch")

    else:
        current_first_lines = text.splitlines()[:6]
        previous_first_lines = previous_text.splitlines()[:6]

        current_start = " ".join(current_first_lines).lower()
        previous_start = " ".join(previous_first_lines).lower()

        # New title/header on current page
        if len(current_first_lines) > 0 and len(current_first_lines[0]) < 80:
            score += 20
            reasons.append("Possible title/header at top of page")

        # Current page has structured identifiers near top
        generic_identifiers = [
            "name", "date", "id", "number", "account", "case",
            "vendor", "employee", "student", "department"
        ]

        for term in generic_identifiers:
            if term in current_start:
                score += 6
                reasons.append(f"Identifier near top: {term}")

        # Current page looks different from prior page
        shared_words = set(lower_text.split()) & set(lower_previous.split())
        current_words = set(lower_text.split())

        if current_words:
            overlap_ratio = len(shared_words) / len(current_words)
        else:
            overlap_ratio = 0

        if overlap_ratio < 0.25:
            score += 25
            reasons.append("Low text similarity to previous page")

        # Continuation signals reduce score
        continuation_terms = [
            "continued", "page 2", "page 3", "page 4",
            "continued on next page", "signature continued"
        ]

        for term in continuation_terms:
            if term in lower_text:
                score -= 35
                reasons.append(f"Continuation signal: {term}")

    confidence = max(0, min(score, 100))
    is_new_document = confidence >= 55

    return {
        "page": page_number,
        "is_new_document": is_new_document,
        "document_type": "Unknown",
        "confidence": confidence,
        "review_needed": confidence < 90,
        "reason": "; ".join(reasons),
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
