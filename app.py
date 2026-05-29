import streamlit as st
import fitz
from PIL import Image
import io
import pandas as pd
from openai import OpenAI
import json

@st.cache_data(show_spinner=False)

# -------------------------
# Process PDF Cache
# -------------------------
def process_pdf_cached(pdf_bytes, active_config):

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

        if page_num + 1 < len(pdf_document):
            next_page = pdf_document[page_num + 1]
            next_page_text = extract_text_from_page(next_page)
        else:
            next_page_text = ""

        result = analyze_page(
            image,
            page_num + 1,
            page_text,
            previous_page_text,
            next_page_text,
            active_config
        )

        results.append(result)

        previous_page_text = page_text

    return results

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------
# FUNCTIONS
# -------------------------

# -------------------------
# Metric Card Helper Function
# -------------------------

def metric_card(icon, label, value):
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 10px;
            background-color: #ffffff;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            min-height: 80px;
            text-align: center;
        ">
            <div style="font-size: 30px;">{icon}</div>
            <div style="font-size: 14px; color: #6b7280;">{label}</div>
            <div style="font-size: 26px; font-weight: 700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
def analyze_page(image, page_number, page_text, previous_page_text="", next_page_text="", project_config=None):

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
- Use next page text to help determine whether the current page is ending a document or whether the next page starts a new document.
- Set review_needed true only when confidence is below the configured threshold.
- For clear document continuations or clear new document starts, confidence may be 90+.
- Use 85–89 for moderately confident decisions that may be auto-approved if the configured threshold allows it.

Project Configuration:
{json.dumps(project_config, indent=2)}

Page Number:
{page_number}

Previous Page Text:
{previous_page_text[:2500]}

Next Page Text:
{next_page_text[:2500]}

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
# Apply Document Statuses
# -------------------------

def apply_document_statuses(documents):

    for doc in documents:

        status = st.session_state.document_statuses.get(
            doc["document_number"]
        )

        if status == "Approved":
            doc["review_needed"] = False

        elif status == "Needs Review":
            doc["review_needed"] = True

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
    padding-top: 3rem;
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

if "document_statuses" not in st.session_state:
    st.session_state.document_statuses = {}

if "selected_config_name" not in st.session_state:
    st.session_state.selected_config_name = "ViaTRON Demo"

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "gallery"

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

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

    st.subheader("Customer Setup")

    config_action = st.radio(
        "What do you want to do?",
        [
            "Use Existing Configuration",
            "Create New Configuration"
        ],
        horizontal=True
    )

    if config_action == "Use Existing Configuration":

        selected_config_name = st.selectbox(
            "Select Customer Configuration",
            list(st.session_state.customer_configs.keys())
        )

        st.session_state.selected_config_name = selected_config_name

        active_preview_config = st.session_state.customer_configs[
            selected_config_name
        ]

        st.success(
            f"Using configuration: {active_preview_config['customer_name']}"
        )

        with st.expander("View Configuration Details"):
            st.json(active_preview_config)

    else:

        st.markdown("### Create Customer Configuration")

        customer_name = st.text_input(
            "Customer / Project Name",
            placeholder="Example: ABC Manufacturing"
        )

        industry = st.selectbox(
            "Industry",
            [
                "Manufacturing",
                "Healthcare",
                "Insurance",
                "Banking / Financial Services",
                "Legal",
                "Government",
                "Education",
                "Logistics / Transportation",
                "Shared Services / BPO",
                "Other"
            ]
        )

        document_types_input = st.text_area(
            "Document Types",
            "Invoice\nPurchase Order\nContract\nPacking Slip\nApplication"
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            50,
            100,
            90
        )

        st.markdown("### Metadata Fields")

        metadata_input = st.text_area(
            "Metadata Fields by Document Type",
            "Invoice: invoice_number, vendor_name, invoice_date, amount\nPurchase Order: po_number, vendor_name, order_date, amount\nContract: contract_id, effective_date, counterparty"
        )

        st.markdown("### Optional Guidance")

        keyword_guidance = st.text_area(
            "Expected Keywords / Notes",
            "Invoice documents often include invoice number, remit to, amount due, vendor name.\nPurchase orders often include PO number, buyer, supplier, order date."
        )

        if st.button("Save Customer Configuration", type="primary"):

            document_types = [
                item.strip()
                for item in document_types_input.splitlines()
                if item.strip()
            ]

            metadata_fields = {}

            for line in metadata_input.splitlines():

                if ":" in line:

                    doc_type, fields = line.split(":", 1)

                    metadata_fields[doc_type.strip()] = [
                        field.strip()
                        for field in fields.split(",")
                        if field.strip()
                    ]

            st.session_state.customer_configs[customer_name] = {
                "customer_name": customer_name,
                "industry": industry,
                "document_types": document_types,
                "confidence_threshold": confidence_threshold,
                "metadata_fields": metadata_fields,
                "keyword_guidance": keyword_guidance
            }

            st.session_state.selected_config_name = customer_name

            st.success(
                f"Configuration saved for {customer_name}."
            )

    st.divider()

    st.subheader("Active Configuration")

    active_config_preview = st.session_state.customer_configs[
        st.session_state.selected_config_name
    ]

    card1, card2, card3 = st.columns(3)

    with card1:
        metric_card(
            "👥",
            "Active Customer",
            active_config_preview["customer_name"]
        )
    
    with card2:
        metric_card(
            "📄",
            "Document Types",
            len(active_config_preview.get("document_types", []))
        )
    
    with card3:
        metric_card(
            "🎯",
            "Threshold",
            f"{active_config_preview.get('confidence_threshold', 90)}%"
        )

    with st.expander("Full Active Configuration"):
        st.json(active_config_preview)

# ==================================================
# DOCUMENT PROCESSING TAB
# ==================================================

with main_tab:

    active_config = (
        st.session_state.customer_configs[
            st.session_state.selected_config_name
        ]
    )

    st.subheader("Document Processing")

    card1, card2, card3 = st.columns(3)

    with card1:
        metric_card(
            "👥",
            "Active Customer",
            active_config["customer_name"]
        )
    
    with card2:
        metric_card(
            "📄",
            "Document Types",
            len(active_config.get("document_types", []))
        )
    
    with card3:
        metric_card(
            "🎯",
            "Threshold",
            f"{active_config.get('confidence_threshold', 90)}%"
        )
    
    st.info(
        "This configuration will be used for document segmentation and classification."
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
                results = process_pdf_cached(pdf_bytes, active_config)
        
                st.write("Grouping documents...")
                documents = apply_document_statuses(
                    group_documents(results)
                )

                for doc in documents:

                    st.write(st.session_state.document_statuses)

                    if doc["document_number"] in st.session_state.document_statuses:
                
                        status = st.session_state.document_statuses[
                            doc["document_number"]
                        ]
                
                        if status == "Approved":
                            doc["review_needed"] = False
                
                        elif status == "Needs Review":
                            doc["review_needed"] = True

                st.session_state.pdf_bytes = pdf_bytes
                st.session_state.analysis_results = results
                st.session_state.total_pages = total_pages
        
                status.update(
                    label="Analysis complete.",
                    state="complete",
                    expanded=False
                )

        if st.session_state.analysis_results is not None:

                results = st.session_state.analysis_results
                pdf_bytes = st.session_state.pdf_bytes
            
                pdf_document = fitz.open(
                    stream=pdf_bytes,
                    filetype="pdf"
                )
            
                documents = apply_document_statuses(
                    group_documents(results)
                )

                for doc in documents:
                    
                    if doc["document_number"] in st.session_state.document_statuses:
                
                        status = st.session_state.document_statuses[
                            doc["document_number"]
                        ]
                
                        if status == "Approved":
                            doc["review_needed"] = False
                
                        elif status == "Needs Review":
                            doc["review_needed"] = True
        
                st.success(
                    f"PDF loaded successfully ({st.session_state.total_pages} pages)"
                )
                st.subheader("Batch Summary")
            
                df = pd.DataFrame(results)
            
                total_pages = len(df)
                avg_confidence = round(df["confidence"].mean(), 1)
                needs_review = len([
                    doc for doc in documents
                    if doc["review_needed"]
                ])
                
                auto_approved = len(documents) - needs_review
                
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
            
                documents = apply_document_statuses(
                    group_documents(results)
                )

                for doc in documents:
                    
                    if doc["document_number"] in st.session_state.document_statuses:
                
                        status = st.session_state.document_statuses[
                            doc["document_number"]
                        ]
                
                        if status == "Approved":
                            doc["review_needed"] = False
                
                        elif status == "Needs Review":
                            doc["review_needed"] = True
            
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
            
                    page_cols = st.columns(2)
            
                    for page_num in range(
                        selected_doc["start_page"] - 1,
                        selected_doc["end_page"]
                    ):
            
                        with page_cols[
                            page_num % 2
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
                        
                            st.session_state.document_statuses[
                                selected_doc["document_number"]
                            ] = "Approved"
                        
                            for result in st.session_state.analysis_results:
                        
                                if (
                                    result["page"] >= selected_doc["start_page"]
                                    and
                                    result["page"] <= selected_doc["end_page"]
                                ):
                                    result["review_needed"] = False
                        
                            st.success("Document approved.")
                            st.rerun()
            
                    with action_col2:
            
                        if st.button(
                            "Needs Review",
                            key=f"review_selected_{selected_doc['document_number']}"
                        ):
                        
                            st.session_state.document_statuses[
                                selected_doc["document_number"]
                            ] = "Needs Review"
                        
                            st.warning("Document marked for review.")
                            st.rerun()
            
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
