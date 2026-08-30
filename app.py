"""
AI-Based Fake Identity & Document Screening System
====================================================
Streamlit frontend that ties together document preprocessing,
OCR extraction, and facial verification into a polished UI.

Run with:  streamlit run app.py
"""

import streamlit as st
import traceback

# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Identity Screening",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from utils.helpers import bytes_to_cv2, cv2_to_pil, draw_face_bbox, resize_for_display
    from modules.document_processor import process_document
    from modules.ocr_extractor import extract_fields
    from modules.face_verifier import verify_faces
    startup_error = None
except Exception as e:
    startup_error = traceback.format_exc()

# ──────────────────────────────────────────────────────────────────────
# Custom CSS — Warm Minimal Light Theme
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&family=DM+Serif+Display&display=swap');

    /* ── Global ─────────────────────────────────────────────── */
    .stApp {
        background: #faf6f1;
        font-family: 'DM Sans', sans-serif;
    }

    /* Ensure Streamlit main content widgets inherit the font */
    .stApp .stMarkdown,
    .stApp .stText,
    .stApp .stCaption,
    .stApp .main p,
    .stApp .main span,
    .stApp .main div {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Header ─────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem 0.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header .badge {
        display: inline-block;
        background: #c2704022;
        color: #c27040;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        font-weight: 400;
        color: #2d2019;
        margin-bottom: 0.4rem;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: #8a7e72;
        font-size: 1rem;
        font-weight: 400;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Warm Cards ─────────────────────────────────────────── */
    .warm-card {
        background: #ffffff;
        border: 1px solid #e8e0d6;
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(45, 32, 25, 0.04);
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }
    .warm-card:hover {
        box-shadow: 0 6px 24px rgba(45, 32, 25, 0.08);
        transform: translateY(-2px);
    }
    .warm-card h3 {
        font-family: 'DM Serif Display', serif;
        color: #2d2019;
        font-weight: 400;
        font-size: 1.25rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .warm-card h3 .icon-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        font-size: 1rem;
    }
    .icon-ocr { background: #dbeafe; }
    .icon-face { background: #fce7cc; }

    /* ── Extracted Field Rows ───────────────────────────────── */
    .field-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem 0;
        background: #faf6f1;
        border-radius: 14px;
        border-left: 3px solid #c27040;
    }
    .field-label {
        color: #8a7e72;
        font-weight: 500;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .field-value {
        color: #2d2019;
        font-weight: 600;
        font-size: 1rem;
    }
    .field-confidence {
        background: #c2704015;
        color: #c27040;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 8px;
    }

    /* ── Verdict Banners ────────────────────────────────────── */
    .verdict-banner {
        text-align: center;
        padding: 1.8rem 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    .verdict-authentic {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #15803d;
    }
    .verdict-flagged {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #a16207;
    }
    .verdict-rejected {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #b91c1c;
    }

    /* ── Upload Area Polish ─────────────────────────────────── */
    .upload-section {
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .upload-section .upload-icon {
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .upload-section .upload-title {
        color: #2d2019;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.15rem;
    }
    .upload-section .upload-desc {
        color: #a39888;
        font-size: 0.8rem;
    }

    /* ── Confidence Meter ───────────────────────────────────── */
    .confidence-ring {
        text-align: center;
        padding: 1.2rem;
    }
    .confidence-ring .value {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-weight: 400;
        color: #c27040;
    }
    .confidence-ring .label {
        color: #8a7e72;
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }
    .confidence-bar-track {
        width: 100%;
        height: 6px;
        background: #f0e8dd;
        border-radius: 3px;
        margin-top: 0.8rem;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #c27040, #e8a06a);
        transition: width 0.8s ease;
    }

    /* ── Divider ────────────────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 1px solid #e8e0d6;
        margin: 2rem 0;
    }

    /* ── Step Indicator ─────────────────────────────────────── */
    .step-bar {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1.2rem 0 1.8rem;
    }
    .step-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 1rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        background: #f0e8dd;
        color: #8a7e72;
        transition: all 0.3s ease;
    }
    .step-pill.active {
        background: #c27040;
        color: #ffffff;
    }
    .step-pill .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        font-size: 0.7rem;
        font-weight: 700;
        background: rgba(255,255,255,0.25);
    }

    /* ── Summary Stat Chips ─────────────────────────────────── */
    .stat-chip {
        background: #ffffff;
        border: 1px solid #e8e0d6;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-chip .stat-val {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #2d2019;
        margin-bottom: 0.2rem;
    }
    .stat-chip .stat-label {
        color: #8a7e72;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Streamlit overrides ────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #c27040 0%, #d4915f 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 2.5rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 14px rgba(194, 112, 64, 0.25);
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(194, 112, 64, 0.4);
        transform: translateY(-2px);
    }
    .stButton > button:disabled {
        background: #d9cfc3;
        box-shadow: none;
        color: #a39888;
    }

    /* File uploader label */
    div[data-testid="stFileUploader"] > label {
        color: #2d2019 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #d9cfc3;
        border-radius: 16px;
        padding: 1.2rem;
        transition: border-color 0.3s ease, background 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #c27040;
        background: #fdf8f3;
    }

    /* Upload button inside the uploader */
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] button {
        background: #c27040 !important;
        color: transparent !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.2rem !important;
        font-size: 0 !important;
        box-shadow: 0 2px 8px rgba(194, 112, 64, 0.2) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        position: relative !important;
        min-width: 120px !important;
        height: 38px !important;
        overflow: hidden !important;
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] button::after {
        content: "Choose File";
        color: #ffffff;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] button span,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] button p,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] button div {
        visibility: hidden !important;
        font-size: 0 !important;
        line-height: 0 !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] button:hover {
        background: #a85d32 !important;
        box-shadow: 0 4px 12px rgba(194, 112, 64, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* File size / type label */
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #a39888 !important;
        font-size: 0.78rem !important;
    }

    /* Uploaded file name chip */
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        background: #faf6f1 !important;
        border: 1px solid #e8e0d6 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e8e0d6;
        border-radius: 14px;
    }

    /* Metric cards override */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8e0d6;
        border-radius: 16px;
        padding: 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #8a7e72 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #2d2019 !important;
        font-family: 'DM Serif Display', serif !important;
    }

    /* Spinner override */
    .stSpinner > div {
        border-top-color: #c27040 !important;
    }

    /* Toast / info / success overrides */
    .stAlert {
        border-radius: 14px !important;
    }

    /* ── Footer ─────────────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 3rem 1rem 1.5rem;
        color: #b5a99a;
        font-size: 0.75rem;
        letter-spacing: 0.3px;
    }
    .footer a {
        color: #c27040;
        text-decoration: none;
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
        <div class="badge">AI-Powered Verification</div>
        <h1>Identity Document Screening</h1>
        <p>Upload your ID card and a selfie — we'll verify document authenticity and match your identity in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if startup_error:
    st.error("### ⚠️ App Failed to Start (Startup/Import Error)")
    st.markdown("Below is the traceback of the exception encountered during initialization:")
    st.code(startup_error, language="python")
    st.stop()

# Step indicator
st.markdown(
    """
    <div class="step-bar">
        <div class="step-pill active">
            <span class="step-num">1</span> Upload
        </div>
        <div class="step-pill">
            <span class="step-num">2</span> Analyse
        </div>
        <div class="step-pill">
            <span class="step-num">3</span> Results
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# File Uploaders — side by side
# ──────────────────────────────────────────────────────────────────────
col_upload1, col_upload2 = st.columns(2, gap="large")

with col_upload1:
    id_file = st.file_uploader(
        "📄  ID Card / Passport",
        type=["jpg", "jpeg", "png"],
        key="id_upload",

    )
    if id_file:
        st.image(id_file, caption="Uploaded ID Card", use_container_width=True)

with col_upload2:
    selfie_file = st.file_uploader(
        "🤳  Selfie Photo",
        type=["jpg", "jpeg", "png"],
        key="selfie_upload",

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
        "✦  Start Verification",
        disabled=not can_verify,
        use_container_width=True,
    )

if not can_verify and not verify_clicked:
    st.markdown(
        """
        <div style="text-align: center; color: #b5a99a; padding: 2rem; font-size: 0.92rem; line-height: 1.7;">
            Upload both documents above to begin the verification process.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────
# Processing Pipeline
# ──────────────────────────────────────────────────────────────────────
if verify_clicked and can_verify:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Update step indicator to show "Analyse" active
    st.markdown(
        """
        <div class="step-bar">
            <div class="step-pill" style="background:#15803d; color:#fff;">
                <span class="step-num">✓</span> Upload
            </div>
            <div class="step-pill active">
                <span class="step-num">2</span> Analyse
            </div>
            <div class="step-pill">
                <span class="step-num">3</span> Results
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # ── Completed step bar ───────────────────────────────────────────
    st.markdown(
        """
        <div class="step-bar">
            <div class="step-pill" style="background:#15803d; color:#fff;">
                <span class="step-num">✓</span> Upload
            </div>
            <div class="step-pill" style="background:#15803d; color:#fff;">
                <span class="step-num">✓</span> Analyse
            </div>
            <div class="step-pill active">
                <span class="step-num">3</span> Results
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Display Results ──────────────────────────────────────────────
    col_ocr, col_face = st.columns(2, gap="large")

    # ── OCR Results Card ─────────────────────────────────────────────
    with col_ocr:
        st.markdown(
            '<div class="warm-card"><h3><span class="icon-circle icon-ocr">📋</span> Extracted Information</h3>',
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
                    <div class="field-row" style="border-left-color: #b91c1c;">
                        <div>
                            <span class="field-label">{label}</span><br>
                            <span class="field-value" style="color: #b91c1c;">Not Detected</span>
                        </div>
                        <span class="field-confidence" style="background: #fef2f2; color: #b91c1c;">—</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Average OCR confidence
        avg_conf = ocr_results["avg_confidence"]
        avg_pct = avg_conf * 100
        st.markdown(
            f"""
            <div style="margin-top: 1.2rem; padding: 0.8rem 1rem; background: #faf6f1;
                        border-radius: 12px; font-size: 0.85rem;">
                <div style="display: flex; justify-content: space-between; color: #8a7e72; margin-bottom: 0.4rem;">
                    <span>Average OCR Confidence</span>
                    <strong style="color: #c27040;">{avg_pct:.1f}%</strong>
                </div>
                <div class="confidence-bar-track">
                    <div class="confidence-bar-fill" style="width: {avg_pct}%;"></div>
                </div>
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
            '<div class="warm-card"><h3><span class="icon-circle icon-face">👤</span> Face Comparison</h3>',
            unsafe_allow_html=True,
        )

        if face_result["error"]:
            st.error(face_result["error"])
        else:
            face_col1, face_col2 = st.columns(2, gap="small")

            with face_col1:
                st.markdown(
                    '<p style="color: #8a7e72; text-align: center; font-size: 0.8rem; margin-bottom: 0.3rem; font-weight: 500;">ID Card Face</p>',
                    unsafe_allow_html=True,
                )
                if face_result["id_face_crop"] is not None:
                    st.image(
                        cv2_to_pil(face_result["id_face_crop"]),
                        use_container_width=True,
                    )

            with face_col2:
                st.markdown(
                    '<p style="color: #8a7e72; text-align: center; font-size: 0.8rem; margin-bottom: 0.3rem; font-weight: 500;">Selfie</p>',
                    unsafe_allow_html=True,
                )
                selfie_display = resize_for_display(selfie_img, max_width=300)
                st.image(cv2_to_pil(selfie_display), use_container_width=True)

            # Confidence meter
            conf = face_result["confidence"]
            st.markdown(
                f"""
                <div class="confidence-ring">
                    <div class="value">{conf}%</div>
                    <div class="label">Face Match Confidence</div>
                    <div class="confidence-bar-track" style="margin-top: 0.6rem;">
                        <div class="confidence-bar-fill" style="width: {conf}%;"></div>
                    </div>
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
            {verdict_icon} Verification Status: {verdict_text}
            <div style="font-family: 'DM Sans', sans-serif; font-size: 0.85rem; font-weight: 400; margin-top: 0.5rem; opacity: 0.75;">
                {verdict_detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary stat chips
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
    <div class="footer">
        AI Identity Document Screening System · Built with Streamlit, OpenCV, EasyOCR & DeepFace
    </div>
    """,
    unsafe_allow_html=True,
)
