import streamlit as st
import fitz
from PIL import Image
import io
import pandas as pd
from openai import OpenAI
import json

@st.cache_data(show_spinner=False)
def process_pdf_cached(pdf_bytes):

    pdf_document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    results = []
    previous_page_text = ""

    for page_num in range(len(pdf_document)):

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

    return results

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
def analyze_page(image, page_number, page_text, previous_page_text="", project_config=None):

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
  "reason": "Brief explanation",
  "metadata": {{}}
}}

Rules:
- Page 1 is always a new document.
- If unsure, set confidence below 90 and review_needed true.
- Do not hardcode specific document names.
- Infer document type if possible.
- Confidence must be 0 to 100.
- Extract important metadata fields if visible.
- Metadata should vary based on document type.
- Return metadata as key/value JSON.
- Use the Project Configuration to guide document classification.
- Prefer the configured document types when the page matches one of them.
- Extract metadata based on the configured metadata fields when available.
- If the document does not match any configured type, classify as "Unknown".
- Use the configured confidence threshold when deciding review_needed.

Project Configuration:
{json.dumps(project_config, indent=2)}

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
        "metadata": ai_result.get("metadata", {}),
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

st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

[data-testid="stVerticalBlock"] {
    gap: 0.3rem;
}

</style>
""", unsafe_allow_html=True)

st.title("AI Document Segmentation PoC")

# -------------------------
# SESSION STATE
# -------------------------

if "review_actions" not in st.session_state:
    st.session_state.review_actions = {}

if "customer_configs" not in st.session_state:

    st.session_state.customer_configs = {

        "ViaTRON Demo": {

            "customer_name": "ViaTRON Demo",

            "document_types": [
                "Resume",
                "Employment Application",
                "Vendor Invoice",
                "Police Case Report",
                "Student Record"
            ],

            "confidence_threshold": 90,

            "metadata_fields": {
                "Vendor Invoice": [
                    "invoice_number",
                    "vendor_name",
                    "invoice_date",
                    "amount"
                ]
            }
        }
    }

if "selected_config_name" not in st.session_state:
    st.session_state.selected_config_name = "ViaTRON Demo"

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "gallery"

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

# -------------------------
# TABS
# -------------------------

main_tab, config_tab = st.tabs(
    [
        "📄 Document Processing",
        "⚙️ Configuration Management"
    ]
)

# ==================================================
# CONFIGURATION TAB
# ==================================================

with config_tab:

    st.subheader("Customer Configuration")

    config_mode = st.radio(
        "Configuration Mode",
        [
            "Use Existing Configuration",
            "Create New Configuration"
        ],
        horizontal=True
    )

    if config_mode == "Use Existing Configuration":

        selected_config_name = st.selectbox(
            "Select Configuration",
            list(st.session_state.customer_configs.keys())
        )

        st.session_state.selected_config_name = (
            selected_config_name
        )

    else:

        new_customer_name = st.text_input(
            "Customer / Project Name"
        )

        new_document_types = st.text_area(
            "Document Types",
            "Invoice\nPurchase Order\nContract"
        )

        new_threshold = st.slider(
            "Confidence Threshold",
            50,
            100,
            90
        )

        if st.button("Save Configuration"):

            st.session_state.customer_configs[
                new_customer_name
            ] = {

                "customer_name": new_customer_name,

                "document_types": [
                    x.strip()
                    for x in new_document_types.splitlines()
                    if x.strip()
                ],

                "confidence_threshold": new_threshold,

                "metadata_fields": {}
            }

            st.session_state.selected_config_name = (
                new_customer_name
            )

            st.success(
                f"{new_customer_name} saved."
            )

    st.divider()

    st.subheader("Configuration Details")

    st.json(
        st.session_state.customer_configs[
            st.session_state.selected_config_name
        ]
    )

# ==================================================
# DOCUMENT PROCESSING TAB
# ==================================================

with main_tab:

    active_config = (
        st.session_state.customer_configs[
            st.session_state.selected_config_name
        ]
    )

    st.subheader(
        f"Active Configuration: {active_config['customer_name']}"
    )

    if st.button("Clear Analysis Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Re-upload or rerun the file.")

    uploaded_file = st.file_uploader(
        "Upload PDF Batch",
        type=["pdf"]
    )
    
    if uploaded_file:

        run_processing = st.button(
            "🚀 Run Analysis",
            type="primary"
        )

        if run_processing:

            with st.status("Analyzing PDF batch...", expanded=True) as status:

            st.write("Reading uploaded PDF...")
            pdf_bytes = uploaded_file.read()
    
            st.write("Opening PDF...")
            pdf_document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )
    
            total_pages = len(pdf_document)
    
            st.write(f"Processing {total_pages} pages with AI...")
            results = process_pdf_cached(pdf_bytes)
    
            st.write("Grouping documents...")
            documents = group_documents(results)
    
            status.update(
                label="Analysis complete.",
                state="complete",
                expanded=False
            )
    
            pdf_bytes = uploaded_file.read()
        
            pdf_document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )
        
            total_pages = len(pdf_document)
        
            results = process_pdf_cached(pdf_bytes)
        
            st.success(
                f"PDF loaded successfully ({total_pages} pages)"
            )
        
            st.subheader("Batch Summary")
        
            df = pd.DataFrame(results)
        
            total_pages = len(df)
            avg_confidence = round(df["confidence"].mean(), 1)
            needs_review = int(df["review_needed"].sum())
            auto_approved = total_pages - needs_review
            
            col1, col2, col3, col4 = st.columns(4)
        
            col1.metric("Total Pages", total_pages)
            col2.metric("Average Confidence", f"{avg_confidence}%")
            col3.metric("Auto Approved", auto_approved)
            col4.metric("Needs Review", needs_review)
        
            if needs_review > 0:
                st.warning(f"{needs_review} pages need human review before final export.")
            else:
                st.success("All pages cleared confidence threshold.")
        
            with st.expander("View AI Results Table"):
                st.dataframe(df, use_container_width=True)
        
            documents = group_documents(results)
        
            if "selected_doc" not in st.session_state:
                st.session_state.selected_doc = documents[0]["document_number"]
            
            approved_docs = [
                doc for doc in documents
                if not doc["review_needed"]
            ]
            
            review_docs = [
                doc for doc in documents
                if doc["review_needed"]
            ]
    
    
# -------------------------
# REVIEW WORKBENCH
# -------------------------
    
            st.divider()
        
            if st.session_state.selected_doc is None:
                st.session_state.selected_doc = documents[0]["document_number"]
        
            approved_docs = [
                doc for doc in documents
                if not doc["review_needed"]
            ]
        
            review_docs = [
                doc for doc in documents
                if doc["review_needed"]
            ]
        
            if st.session_state.view_mode == "gallery":
        
                st.subheader("Document Gallery")
        
                gallery_filter = st.radio(
                    "Filter",
                    ["All", "Needs Review", "Approved"],
                    horizontal=True,
                    key="gallery_filter"
                )
        
                if gallery_filter == "Needs Review":
                    display_docs = review_docs
                elif gallery_filter == "Approved":
                    display_docs = approved_docs
                else:
                    display_docs = documents
        
                cols = st.columns(8)
        
                for i, doc in enumerate(display_docs):
        
                    with cols[i % 8]:
        
                        with st.container(border=True):
        
                            preview_page = pdf_document[doc["start_page"] - 1]
        
                            preview_pix = preview_page.get_pixmap(
                                matrix=fitz.Matrix(0.35, 0.35)
                            )
        
                            preview_image = Image.open(
                                io.BytesIO(preview_pix.tobytes("png"))
                            )
        
                            st.image(
                                preview_image,
                                width=75
                            )
        
                            st.caption(
                                f"Doc {doc['document_number']}"
                            )
        
                            st.caption(
                                f"{doc['confidence']}%"
                            )
        
                            if doc["review_needed"]:
                                st.warning("Review")
                            else:
                                st.success("OK")
        
                            if st.button(
                                "Open",
                                key=f"open_doc_{doc['document_number']}",
                                use_container_width=True
                            ):
                                st.session_state.selected_doc = doc["document_number"]
                                st.session_state.view_mode = "review"
                                st.rerun()
        
        
            elif st.session_state.view_mode == "review":
        
                selected_doc = next(
                    d for d in documents
                    if d["document_number"] == st.session_state.selected_doc
                )
        
                if st.button("← Back to Gallery"):
                    st.session_state.view_mode = "gallery"
                    st.rerun()
        
                st.subheader(
                    f"Document {selected_doc['document_number']} Review"
                )
        
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
                info_col1.metric(
                    "Type",
                    selected_doc["document_type"]
                )
        
                info_col2.metric(
                    "Pages",
                    f"{selected_doc['start_page']}–{selected_doc['end_page']}"
                )
        
                info_col3.metric(
                    "Confidence",
                    f"{selected_doc['confidence']}%"
                )
        
                if selected_doc["review_needed"]:
                    info_col4.warning("Needs Review")
                else:
                    info_col4.success("Approved")
        
                st.subheader("Pages")
        
                page_cols = st.columns(6)
        
                for page_num in range(
                    selected_doc["start_page"] - 1,
                    selected_doc["end_page"]
                ):
        
                    with page_cols[
                        page_num % 6
                    ]:
        
                        page = pdf_document[page_num]
        
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(0.45, 0.45)
                        )
        
                        img = Image.open(
                            io.BytesIO(
                                pix.tobytes("png")
                            )
                        )
        
                        st.image(
                            img,
                            use_container_width=True
                        )
        
                        st.caption(
                            f"Page {page_num + 1}"
                        )
        
                st.subheader("Actions")
        
                action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
                with action_col1:
        
                    if st.button(
                        "Approve",
                        key=f"approve_selected_{selected_doc['document_number']}"
                    ):
                        st.session_state.review_actions[selected_doc["document_number"]] = "Approved"
                        st.success("Document approved.")
        
                with action_col2:
        
                    if st.button(
                        "Needs Review",
                        key=f"review_selected_{selected_doc['document_number']}"
                    ):
                        st.session_state.review_actions[selected_doc["document_number"]] = "Needs Review"
                        st.warning("Document marked for review.")
        
                with action_col3:
        
                    if st.button(
                        "Rerun AI",
                        key=f"rerun_selected_{selected_doc['document_number']}"
                    ):
                        st.session_state.review_actions[selected_doc["document_number"]] = "Rerun Requested"
                        st.info("Rerun requested.")
        
                with action_col4:
        
                    pdf_bytes_output = create_separated_pdf(
                        pdf_document,
                        selected_doc["start_page"],
                        selected_doc["end_page"]
                    )
        
                    st.download_button(
                        label="Download",
                        data=pdf_bytes_output,
                        file_name=f"Document_{selected_doc['document_number']}_{selected_doc['document_type'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key=f"download_selected_{selected_doc['document_number']}"
                    )
