import streamlit as st
import fitz
from PIL import Image
import io
import pandas as pd
from openai import OpenAI
import json

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------
# FUNCTIONS
# -------------------------

# -------------------------
# Create Seperated PDF
# -------------------------

def create_separated_pdf(pdf_document, start_page, end_page):

    output_pdf = fitz.open()

    for page_index in range(start_page - 1, end_page):
        output_pdf.insert_pdf(
            pdf_document,
            from_page=page_index,
            to_page=page_index
        )

    pdf_bytes = output_pdf.tobytes()
    output_pdf.close()

    return pdf_bytes

# -------------------------
# Clean AI JSON
# -------------------------

def clean_ai_json(content):

    content = content.strip()

    if content.startswith("```json"):
        content = content.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

    elif content.startswith("```"):
        content = content.replace(
            "```",
            ""
        ).strip()

    return json.loads(content)

# -------------------------
# Analyze Page
# -------------------------
def analyze_page(image, page_number, page_text, previous_page_text=""):

    prompt = f"""
You are analyzing scanned business documents for document segmentation.

Your job:
Determine whether the current page starts a new document or continues the previous document.

Use:
- current page text
- previous page text
- document structure
- headers/titles
- changes in topic
- form layout clues
- page continuation clues

Return JSON only.

Schema:
{{
  "is_new_document": true,
  "document_type": "Unknown",
  "confidence": 0,
  "review_needed": true,
  "reason": "Brief explanation"
}}

Rules:
- Page 1 is always a new document.
- If unsure, set confidence below 90 and review_needed true.
- Do not hardcode specific document names.
- Infer document type if possible.
- Confidence must be 0 to 100.

Page Number:
{page_number}

Previous Page Text:
{previous_page_text[:2500]}

Current Page Text:
{page_text[:3500]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    try:
        ai_result = clean_ai_json(content)
    except Exception as e:
        ai_result = {
            "is_new_document": False,
            "document_type": "Unknown",
            "confidence": 0,
            "review_needed": True,
            "reason": f"AI response could not be parsed: {str(e)} | Raw: {content[:300]}"
        }

    return {
        "page": page_number,
        "is_new_document": ai_result.get("is_new_document", False),
        "document_type": ai_result.get("document_type", "Unknown"),
        "confidence": ai_result.get("confidence", 0),
        "review_needed": ai_result.get("review_needed", True),
        "reason": ai_result.get("reason", ""),
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
    previous_page_text = ""

    for page_num in range(total_pages):

        page = pdf_document[page_num]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(1.5, 1.5)
        )

        img_bytes = pix.tobytes("png")

        image = Image.open(
            io.BytesIO(img_bytes)
        )

        page_text = extract_text_from_page(page)

        result = analyze_page(
            image,
            page_num + 1,
            page_text,
            previous_page_text
        )

        results.append(result)

        previous_page_text = page_text

        st.write(result)

        if st.button(
            f"Rerun AI - Page {page_num + 1}",
            key=f"rerun_{page_num}"
        ):
        
            rerun_result = analyze_page(
                image,
                page_num + 1,
                page_text,
                previous_page_text
            )
        
            st.write("Rerun Result:")
            st.write(rerun_result)
        
        st.image(
            image,
            caption=f"Page {page_num + 1}",
            width=300
        )
        
        st.divider()

    st.subheader("AI Segmentation Results")

    df = pd.DataFrame(results)

    st.dataframe(df, use_container_width=True)

    st.subheader("Grouped Documents")

    documents = group_documents(results)

    doc_df = pd.DataFrame(documents)

    st.dataframe(doc_df, use_container_width=True)

    st.subheader("Download Separated Documents")

    for doc in documents:

        pdf_bytes_output = create_separated_pdf(
            pdf_document,
            doc["start_page"],
            doc["end_page"]
        )

        st.download_button(
            label=f"Download Document {doc['document_number']} - {doc['document_type']}",
            data=pdf_bytes_output,
            file_name=f"Document_{doc['document_number']}_{doc['document_type'].replace(' ', '_')}.pdf",
            mime="application/pdf",
            key=f"download_{doc['document_number']}"
        )
