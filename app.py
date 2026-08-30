"""
AI-Based Fake Identity & Document Screening System
====================================================
Streamlit frontend that ties together document preprocessing,
OCR extraction, and facial verification into a polished UI.

Run with:  streamlit run app.py
"""

import streamlit as st
from utils.helpers import bytes_to_cv2, cv2_to_pil, draw_face_bbox, resize_for_display
from modules.document_processor import process_document
from modules.ocr_extractor import extract_fields
from modules.face_verifier import verify_faces

# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Identity Screening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS — Premium dark-themed styling
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(145deg, #0a0e1a 0%, #111827 40%, #0d1321 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ─────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 300;
    }

    /* ── Glass Cards ────────────────────────────────────────── */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.08);
    }
    .glass-card h3 {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Extracted Field Rows ───────────────────────────────── */
    .field-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        border-left: 3px solid #6366f1;
    }
    .field-label {
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .field-value {
        color: #f1f5f9;
        font-weight: 600;
        font-size: 1rem;
    }
    .field-confidence {
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 400;
    }

    /* ── Verdict Banners ────────────────────────────────────── */
    .verdict-banner {
        text-align: center;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .verdict-authentic {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(52, 211, 153, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
    }
    .verdict-flagged {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(251, 191, 36, 0.08) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fbbf24;
    }
    .verdict-rejected {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(248, 113, 113, 0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
    }

    /* ── Upload Area Polish ─────────────────────────────────── */
    .upload-label {
        color: #cbd5e1;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    /* ── Confidence Meter ───────────────────────────────────── */
    .confidence-meter {
        text-align: center;
        padding: 1rem;
    }
    .confidence-meter .value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .confidence-meter .label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* ── Divider ────────────────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(99, 102, 241, 0.12);
        margin: 2rem 0;
    }

    /* ── Streamlit overrides ────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
    }

    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ AI Identity Document Screening</h1>
        <p>Upload an ID card and a selfie to verify document authenticity</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# File Uploaders — side by side
# ──────────────────────────────────────────────────────────────────────
col_upload1, col_upload2 = st.columns(2, gap="large")

with col_upload1:
    st.markdown('<p class="upload-label">📄 Upload ID Card / Passport</p>', unsafe_allow_html=True)
    id_file = st.file_uploader(
        "Upload ID Card",
        type=["jpg", "jpeg", "png"],
        key="id_upload",
        label_visibility="collapsed",
    )
    if id_file:
        st.image(id_file, caption="Uploaded ID Card", use_container_width=True)

with col_upload2:
    st.markdown('<p class="upload-label">🤳 Upload Selfie Photo</p>', unsafe_allow_html=True)
    selfie_file = st.file_uploader(
        "Upload Selfie",
        type=["jpg", "jpeg", "png"],
        key="selfie_upload",
        label_visibility="collapsed",
    )
    if selfie_file:
        st.image(selfie_file, caption="Uploaded Selfie", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# Verification Button
# ──────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

can_verify = id_file is not None and selfie_file is not None

col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
with col_btn_center:
    verify_clicked = st.button(
        "🔍  Run Verification",
        disabled=not can_verify,
        use_container_width=True,
    )

if not can_verify and not verify_clicked:
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; padding: 2rem; font-size: 0.95rem;">
            Upload both an ID card and a selfie to begin verification.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────
# Processing Pipeline
# ──────────────────────────────────────────────────────────────────────
if verify_clicked and can_verify:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 1: Load images ──────────────────────────────────────────
    with st.spinner("Loading and preprocessing document..."):
        try:
            id_img = bytes_to_cv2(id_file)
            selfie_img = bytes_to_cv2(selfie_file)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()

        doc = process_document(id_img)

    # ── Step 2: OCR Extraction ───────────────────────────────────────
    with st.spinner("Extracting text from document (this may take a moment)..."):
        ocr_results = extract_fields(doc["enhanced"])

    # ── Step 3: Face Verification ────────────────────────────────────
    with st.spinner("Verifying faces..."):
        face_result = verify_faces(doc["color"], selfie_img)

    # ── Display Results ──────────────────────────────────────────────
    col_ocr, col_face = st.columns(2, gap="large")

    # ── OCR Results Card ─────────────────────────────────────────────
    with col_ocr:
        st.markdown(
            '<div class="glass-card"><h3>📋 Extracted Information</h3>',
            unsafe_allow_html=True,
        )

        fields = [
            ("Name", ocr_results["name"]),
            ("Date of Birth", ocr_results["date_of_birth"]),
            ("ID Number", ocr_results["id_number"]),
        ]

        for label, (value, confidence) in fields:
            if value:
                conf_pct = f"{confidence * 100:.1f}%"
                st.markdown(
                    f"""
                    <div class="field-row">
                        <div>
                            <span class="field-label">{label}</span><br>
                            <span class="field-value">{value}</span>
                        </div>
                        <span class="field-confidence">{conf_pct}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="field-row" style="border-left-color: #ef4444;">
                        <div>
                            <span class="field-label">{label}</span><br>
                            <span class="field-value" style="color: #f87171;">Not Detected</span>
                        </div>
                        <span class="field-confidence" style="color: #ef4444;">—</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Average OCR confidence
        avg_conf = ocr_results["avg_confidence"]
        st.markdown(
            f"""
            <div style="margin-top: 1rem; padding: 0.5rem 1rem; background: rgba(15, 23, 42, 0.4);
                        border-radius: 8px; color: #94a3b8; font-size: 0.85rem;">
                Average OCR Confidence: <strong style="color: #a78bfa;">{avg_conf * 100:.1f}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Raw OCR output in expander
        with st.expander("🔬 Raw OCR Output"):
            st.text(ocr_results["raw_text"])

    # ── Face Comparison Card ─────────────────────────────────────────
    with col_face:
        st.markdown(
            '<div class="glass-card"><h3>👤 Face Comparison</h3>',
            unsafe_allow_html=True,
        )

        if face_result["error"]:
            st.error(face_result["error"])
        else:
            face_col1, face_col2 = st.columns(2, gap="small")

            with face_col1:
                st.markdown(
                    '<p style="color: #94a3b8; text-align: center; font-size: 0.8rem; margin-bottom: 0.3rem;">ID Card Face</p>',
                    unsafe_allow_html=True,
                )
                if face_result["id_face_crop"] is not None:
                    st.image(
                        cv2_to_pil(face_result["id_face_crop"]),
                        use_container_width=True,
                    )

            with face_col2:
                st.markdown(
                    '<p style="color: #94a3b8; text-align: center; font-size: 0.8rem; margin-bottom: 0.3rem;">Selfie</p>',
                    unsafe_allow_html=True,
                )
                selfie_display = resize_for_display(selfie_img, max_width=300)
                st.image(cv2_to_pil(selfie_display), use_container_width=True)

            # Confidence meter
            conf = face_result["confidence"]
            st.markdown(
                f"""
                <div class="confidence-meter">
                    <div class="value">{conf}%</div>
                    <div class="label">Face Match Confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # Final Verdict
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Determine verdict
    face_ok = face_result.get("verified", False)
    face_error = face_result.get("error") is not None
    ocr_name = ocr_results["name"][0] is not None
    ocr_dob = ocr_results["date_of_birth"][0] is not None
    ocr_id = ocr_results["id_number"][0] is not None
    ocr_fields_found = sum([ocr_name, ocr_dob, ocr_id])
    avg_ocr_conf = ocr_results["avg_confidence"]

    if face_ok and ocr_fields_found >= 2 and avg_ocr_conf >= 0.6:
        verdict_class = "verdict-authentic"
        verdict_icon = "✅"
        verdict_text = "AUTHENTIC"
        verdict_detail = "Document passed all verification checks."
    elif face_error:
        verdict_class = "verdict-rejected"
        verdict_icon = "❌"
        verdict_text = "FLAGGED FOR REVIEW"
        verdict_detail = "Face could not be detected or compared."
    elif not face_ok:
        verdict_class = "verdict-flagged"
        verdict_icon = "⚠️"
        verdict_text = "FLAGGED FOR REVIEW"
        verdict_detail = "Face mismatch detected between ID card and selfie."
    else:
        verdict_class = "verdict-flagged"
        verdict_icon = "⚠️"
        verdict_text = "FLAGGED FOR REVIEW"
        verdict_detail = "Some document fields could not be verified."

    st.markdown(
        f"""
        <div class="verdict-banner {verdict_class}">
            {verdict_icon} VERIFICATION STATUS: {verdict_text}
            <div style="font-size: 0.85rem; font-weight: 400; margin-top: 0.5rem; opacity: 0.8;">
                {verdict_detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Face Match", f"{face_result['confidence']}%")
    m2.metric("OCR Fields Found", f"{ocr_fields_found} / 3")
    m3.metric("Avg OCR Confidence", f"{avg_ocr_conf * 100:.1f}%")
    m4.metric("Verdict", verdict_text)

# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 3rem 1rem 1rem; color: #475569; font-size: 0.75rem;">
        AI Identity Document Screening System • Built with Streamlit, OpenCV, EasyOCR & DeepFace
    </div>
    """,
    unsafe_allow_html=True,
)
