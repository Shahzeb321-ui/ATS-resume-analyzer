from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from docx import Document
from pypdf import PdfReader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "resume-fit-final"
LABEL_ORDER = ("Good Fit", "Potential Fit", "No Fit")
FIT_COLORS = {
    "Good Fit": "#5a9e1b",
    "Potential Fit": "#c27a16",
    "No Fit": "#e5484d",
}

st.set_page_config(
    page_title="Resume fit predictor",
    page_icon="📄",
    layout="centered",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17231d;
        --muted: #66756c;
        --green: #356b43;
        --green-dark: #234d32;
        --line: #e5eae5;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(224, 239, 218, .72), transparent 29rem),
            linear-gradient(180deg, #f7faf6 0%, #fff 44rem);
    }
    .block-container {
        max-width: 820px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .hero {
        margin: 0 auto 1.5rem;
        max-width: 700px;
        text-align: center;
    }
    .hero-badge {
        background: #e8f2e7;
        border: 1px solid #d7e7d5;
        border-radius: 999px;
        color: var(--green-dark);
        display: inline-block;
        font-size: .78rem;
        font-weight: 650;
        letter-spacing: .035em;
        margin-bottom: .9rem;
        padding: .38rem .8rem;
    }
    .hero h1 {
        color: var(--ink);
        font-size: clamp(2.1rem, 5vw, 3.1rem);
        font-weight: 750;
        letter-spacing: -.055em;
        line-height: 1.08;
        margin: 0;
    }
    .hero h1 span {
        color: var(--green);
    }
    .app-subtitle {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
        margin: .8rem auto 0;
        max-width: 520px;
    }
    .input-card,
    .results-card {
        background: rgba(255, 255, 255, .92);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 12px 36px rgba(29, 48, 34, .055);
        padding: clamp(1rem, 3vw, 1.6rem);
    }
    .input-intro {
        margin-bottom: 1.2rem;
    }
    .section-kicker {
        color: var(--green);
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .section-title {
        color: var(--ink);
        font-size: 1.22rem;
        font-weight: 700;
        margin: .2rem 0 0;
    }
    .section-copy {
        color: var(--muted);
        font-size: .88rem;
        margin: .25rem 0 0;
    }
    div[data-testid="stFileUploader"] section {
        background: #f8faf7;
        border: 1px dashed #b8c9b9;
        border-radius: 12px;
        transition: background .18s ease, border-color .18s ease;
    }
    div[data-testid="stFileUploader"] section:hover {
        background: #f1f7ef;
        border-color: #6b9573;
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--muted);
    }
    div[data-testid="stTextArea"] textarea {
        background: #fbfcfa;
        border: 1px solid #dce4dc;
        border-radius: 12px;
        line-height: 1.55;
        transition: border-color .18s ease, box-shadow .18s ease;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #6b9573;
        box-shadow: 0 0 0 3px rgba(83, 135, 92, .12);
    }
    div[data-testid="stFileUploader"] label,
    div[data-testid="stTextArea"] label {
        color: #34483a;
        font-weight: 600;
    }
    .format-hint {
        color: var(--muted);
        font-size: .78rem;
        margin: -.35rem 0 1rem;
    }
    .result-card {
        min-height: 114px;
        padding: 19px;
        border: 1px solid transparent;
        border-radius: 14px;
    }
    .fit-card {
        background: linear-gradient(135deg, #edf6e6, #e6f1df);
        border-color: #dbe9d1;
    }
    .confidence-card {
        background: linear-gradient(135deg, #f8f8f5, #f1f3ef);
        border-color: #e9ece7;
    }
    .result-label {
        color: #55665a;
        font-size: .82rem;
        font-weight: 550;
        margin-bottom: 8px;
    }
    .result-value {
        color: #285331;
        font-size: 1.85rem;
        font-weight: 750;
        line-height: 1.2;
    }
    .confidence-card .result-value {
        color: var(--ink);
    }
    .probability-heading {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
        margin: 1.65rem 0 .85rem;
    }
    .probability-row {
        align-items: center;
        display: grid;
        gap: 14px;
        grid-template-columns: 102px minmax(50px, 1fr) 40px;
        margin: 15px 0;
    }
    .probability-label,
    .probability-value {
        color: #46564a;
        font-size: .83rem;
        font-weight: 550;
    }
    .probability-value {
        text-align: right;
    }
    .probability-track {
        background: #edf0eb;
        border-radius: 99px;
        height: 9px;
        overflow: hidden;
    }
    .probability-fill {
        border-radius: inherit;
        height: 100%;
        min-width: 4px;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(105deg, #28543a, #386d49);
        border: 0;
        border-radius: 11px;
        box-shadow: 0 5px 14px rgba(41, 84, 57, .18);
        font-weight: 650;
        min-height: 48px;
        transition: box-shadow .18s ease, transform .18s ease;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(105deg, #214831, #2e5e3e);
        box-shadow: 0 7px 18px rgba(41, 84, 57, .25);
        transform: translateY(-1px);
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    @media (max-width: 520px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.75rem;
        }
        .probability-row {
            gap: 8px;
            grid-template-columns: 88px minmax(40px, 1fr) 36px;
        }
        .result-card {
            min-height: 98px;
            padding: 14px;
        }
        .result-value {
            font-size: 1.55rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_dir: str):
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def extract_resume_text(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    extension = Path(uploaded_file.name).suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if extension == ".docx":
        document = Document(BytesIO(file_bytes))
        parts = [paragraph.text for paragraph in document.paragraphs]
        parts.extend(
            " ".join(cell.text for cell in row.cells)
            for table in document.tables
            for row in table.rows
        )
        return "\n".join(part for part in parts if part.strip()).strip()

    return file_bytes.decode("utf-8-sig").strip()


st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">✦ SMART CAREER MATCHING</div>
        <h1>Find your <span>fit.</span></h1>
        <p class="app-subtitle">See how your experience lines up with the role. Add a resume and job description to get a personalized fit prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        """
        <div class="input-intro">
            <div class="section-kicker">01 &nbsp;·&nbsp; Your details</div>
            <div class="section-title">Bring your next role into focus</div>
            <p class="section-copy">Your resume and the job posting are compared privately on this device.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_resume = st.file_uploader(
        "Upload resume",
        type=["pdf", "docx", "txt"],
        help="Supported file types: PDF, DOCX, and TXT.",
    )
    st.markdown(
        '<p class="format-hint">PDF &nbsp;·&nbsp; DOCX &nbsp;·&nbsp; TXT</p>',
        unsafe_allow_html=True,
    )
    job_description = st.text_area(
        "Job description",
        placeholder="Paste the job description here...",
        height=130,
    )

    analyze_clicked = st.button(
        "✦  Analyze my fit",
        type="primary",
        use_container_width=True,
    )

if analyze_clicked:
    if uploaded_resume is None:
        st.warning("Upload a resume before analyzing fit.")
    elif not job_description.strip():
        st.warning("Add a job description before analyzing fit.")
    else:
        resume_text = extract_resume_text(uploaded_resume)
        if not resume_text:
            st.error(
                "No readable text was found in this resume. "
                "For scanned PDFs, use a text-based PDF or upload a DOCX/TXT file."
            )
        else:
            with st.spinner("Analyzing resume and job description..."):
                tokenizer, model = load_model(str(MODEL_DIR))
                inputs = tokenizer(
                    resume_text,
                    job_description.strip(),
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.inference_mode():
                    logits = model(**inputs).logits[0]
                    probabilities = torch.softmax(logits, dim=-1).tolist()

            labels = model.config.id2label
            scores = {
                labels[index]: probability
                for index, probability in enumerate(probabilities)
            }
            predicted_label = max(scores, key=scores.get)
            st.session_state["resume_fit_result"] = (
                predicted_label,
                scores,
            )

result = st.session_state.get("resume_fit_result")
if result is not None:
    predicted_label, scores = result
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="input-intro">
                <div class="section-kicker">02 &nbsp;·&nbsp; Your results</div>
                <div class="section-title">Here’s how it looks</div>
                <p class="section-copy">A snapshot of how closely your resume matches this opportunity.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fit_column, confidence_column = st.columns(2)
        fit_column.markdown(
            f"""
            <div class="result-card fit-card">
                <div class="result-label">Predicted fit</div>
                <div class="result-value">{predicted_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        confidence_column.markdown(
            f"""
            <div class="result-card confidence-card">
                <div class="result-label">Confidence</div>
                <div class="result-value">{scores[predicted_label]:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="probability-heading">Class probabilities</div>',
            unsafe_allow_html=True,
        )
        for label in LABEL_ORDER:
            probability = scores[label]
            percent = probability * 100
            st.markdown(
                f"""
                <div class="probability-row">
                    <span class="probability-label">{label}</span>
                    <div class="probability-track">
                        <div class="probability-fill"
                             style="width: {percent:.2f}%; background: {FIT_COLORS[label]}"></div>
                    </div>
                    <span class="probability-value">{percent:.0f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
