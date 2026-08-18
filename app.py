import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv(".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Research Paper Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background: #f7f9fc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #111827 0%, #1e3a8a 55%, #312e81 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.18);
        }

        .hero h1 {
            font-size: 2.35rem;
            margin-bottom: 0.4rem;
            font-weight: 800;
        }

        .hero p {
            font-size: 1.02rem;
            opacity: 0.9;
            margin-bottom: 0;
        }

        .badge-row {
            margin-top: 1rem;
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            margin-right: 0.4rem;
            margin-bottom: 0.3rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.13);
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .section-card {
            padding: 1.25rem 1.4rem;
            border-radius: 17px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .step-label {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #4f46e5;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }

        .step-title {
            font-size: 1.45rem;
            font-weight: 750;
            color: #111827;
            margin-bottom: 0.35rem;
        }

        .step-description {
            color: #6b7280;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }

        .pipeline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.35rem;
            margin: 1rem 0 1.6rem 0;
            padding: 1rem;
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 17px;
        }

        .pipeline-item {
            text-align: center;
            flex: 1;
            font-size: 0.78rem;
            color: #374151;
            font-weight: 700;
        }

        .pipeline-icon {
            font-size: 1.45rem;
            display: block;
            margin-bottom: 0.25rem;
        }

        .arrow {
            color: #9ca3af;
            font-size: 1.1rem;
        }

        .result-supported {
            padding: 0.85rem 1rem;
            border-radius: 12px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
            font-weight: 800;
        }

        .result-questionable {
            padding: 0.85rem 1rem;
            border-radius: 12px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
            font-weight: 800;
        }

        .result-contradicted {
            padding: 0.85rem 1rem;
            border-radius: 12px;
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            font-weight: 800;
        }

        .info-box {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e40af;
            font-size: 0.9rem;
        }

        footer {
            visibility: hidden;
        }

        div[data-testid="stFileUploader"] {
            background: white;
            padding: 0.7rem;
            border-radius: 15px;
            border: 1px dashed #9ca3af;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. SESSION STATE
# ============================================================

DEFAULTS = {
    "claims": "",
    "claim_list": [],
    "evidence": [],
    "selected_claim": "",
    "verification_result": "",
    "overall_report": "",
    "paper_signature": None,
    "vectorstore": None,
    "paper_text": "",
    "page_count": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 5. LLM
# ============================================================

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY was not found in your .env file.")
    st.stop()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY,
)


# ============================================================
# 6. HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div style="font-size:0.85rem; font-weight:700; opacity:0.8;">
            INTELLIGENT RESEARCH ANALYSIS PLATFORM
        </div>
        <h1>🔬 AI-Powered Research Paper Verification</h1>
        <p>
            An intelligent GenAI + RAG + Agentic AI workflow for extracting
            research claims, retrieving evidence, verifying claims, and
            generating an evidence-aware paper assessment.
        </p>
        <div class="badge-row">
            <span class="badge">✨ Generative AI</span>
            <span class="badge">🤖 Agentic AI</span>
            <span class="badge">🔎 RAG</span>
            <span class="badge">🧠 NLP</span>
            <span class="badge">📚 FAISS</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. PIPELINE
# ============================================================

st.markdown(
    """
    <div class="pipeline">
        <div class="pipeline-item">
            <span class="pipeline-icon">📄</span>
            Upload Paper
        </div>
        <div class="arrow">→</div>
        <div class="pipeline-item">
            <span class="pipeline-icon">🧠</span>
            Claim Extraction
        </div>
        <div class="arrow">→</div>
        <div class="pipeline-item">
            <span class="pipeline-icon">🔎</span>
            RAG Retrieval
        </div>
        <div class="arrow">→</div>
        <div class="pipeline-item">
            <span class="pipeline-icon">🤖</span>
            AI Verification
        </div>
        <div class="arrow">→</div>
        <div class="pipeline-item">
            <span class="pipeline-icon">📊</span>
            Overall Analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🔬 Research Intelligence")
    st.caption("GenAI • RAG • Agentic AI")

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        """
        **1. Upload**  
        Load a research paper in PDF format.

        **2. Extract**  
        AI identifies important research claims.

        **3. Retrieve**  
        RAG searches the paper for relevant evidence.

        **4. Verify**  
        The verification agent evaluates the selected claim.

        **5. Analyze**  
        Generate an evidence-aware overall assessment.
        """
    )

    st.divider()

    st.markdown("### Technology Stack")
    st.caption("• Groq / Llama 3.1")
    st.caption("• LangChain")
    st.caption("• Hugging Face Embeddings")
    st.caption("• FAISS Vector Search")
    st.caption("• PyPDF")
    st.caption("• Streamlit")

    st.divider()

    st.caption(
        "⚠️ This tool assists verification. "
        "It does not establish scientific truth."
    )


# ============================================================
# 9. UPLOAD
# ============================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.markdown(
    '<div class="step-label">INPUT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="step-title">📄 Upload Research Paper</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="step-description">'
    "Upload a text-based research paper in PDF format to begin the analysis."
    "</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    label_visibility="collapsed",
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 10. PROCESS PAPER
# ============================================================

if uploaded_file is not None:

    file_signature = f"{uploaded_file.name}-{uploaded_file.size}"

    # Reset analysis only when a new file is selected.
    if st.session_state["paper_signature"] != file_signature:
        st.session_state["paper_signature"] = file_signature
        st.session_state["claims"] = ""
        st.session_state["claim_list"] = []
        st.session_state["evidence"] = []
        st.session_state["selected_claim"] = ""
        st.session_state["verification_result"] = ""
        st.session_state["overall_report"] = ""
        st.session_state["vectorstore"] = None
        st.session_state["paper_text"] = ""
        st.session_state["page_count"] = 0

    try:
        reader = PdfReader(uploaded_file)
        pages_text = [
            page.extract_text() or ""
            for page in reader.pages
        ]
        text = "\n".join(pages_text).strip()

    except Exception as exc:
        st.error(f"❌ Could not read this PDF: {exc}")
        st.stop()

    if not text:
        st.error(
            "❌ No readable text was found in this PDF. "
            "The PDF may contain scanned images instead of text."
        )
        st.stop()

    st.session_state["paper_text"] = text
    st.session_state["page_count"] = len(reader.pages)

    st.success("✅ Research paper loaded successfully.")

    # --------------------------------------------------------
    # PAPER METRICS
    # --------------------------------------------------------

    st.markdown("### 📈 Paper Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Pages", len(reader.pages))

    with c2:
        st.metric("Characters", f"{len(text):,}")

    # --------------------------------------------------------
    # CHUNKING + VECTOR STORE
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = splitter.split_text(text)

    with c3:
        st.metric("Evidence Chunks", len(chunks))

    if st.session_state["vectorstore"] is None:
        with st.spinner("Preparing semantic search..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            st.session_state["vectorstore"] = FAISS.from_texts(
                chunks,
                embeddings,
            )

    vectorstore = st.session_state["vectorstore"]

    with c4:
        st.metric("Retrieval", "Ready")

    with st.expander("📖 View Extracted Text"):
        st.write(text[:10000])

    st.divider()


    # ========================================================
    # STEP 1 — CLAIM EXTRACTION
    # ========================================================

    st.markdown(
        '<div class="step-label">STEP 1</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-title">🧠 AI Claim Extraction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-description">'
        "The GenAI agent identifies important claims that can be "
        "evaluated against evidence contained in the paper."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "✨ Extract Important Claims",
        use_container_width=True,
    ):
        claim_prompt = f"""
You are an AI research paper claim extraction agent.

Analyze the research paper below.

Identify exactly 5 important claims that can be checked against
evidence contained within the paper.

Prioritize:
- scientific claims
- experimental findings
- numerical results
- performance claims
- comparison claims
- conclusions

For each claim provide:

CLAIM:
The exact or closely paraphrased claim.

IMPORTANCE:
Why this claim matters.

Do not invent information.

RESEARCH PAPER:
{text[:16000]}
"""

        with st.spinner(
            "AI Claim Extraction Agent is analyzing the paper..."
        ):
            try:
                response = llm.invoke(claim_prompt)
                claims_text = response.content

                st.session_state["claims"] = claims_text

                # Extract lines beginning with "CLAIM:".
                extracted = []
                lines = claims_text.splitlines()

                for line in lines:
                    stripped = line.strip()
                    if stripped.upper().startswith("CLAIM:"):
                        claim_value = stripped.split(":", 1)[1].strip()
                        if claim_value:
                            extracted.append(claim_value)

                st.session_state["claim_list"] = extracted[:5]

                st.success("✅ Claim extraction completed.")

            except Exception as exc:
                st.error(f"❌ Claim extraction failed: {exc}")


    if st.session_state["claims"]:
        st.markdown("### 📋 Extracted Research Claims")
        st.write(st.session_state["claims"])

        st.markdown(
            '<div class="info-box">'
            "💡 Select a claim below to send it to the RAG evidence retrieval stage."
            "</div>",
            unsafe_allow_html=True,
        )


    # ========================================================
    # STEP 2 — EVIDENCE RETRIEVAL / RAG
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="step-label">STEP 2</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-title">🔎 Evidence Retrieval (RAG)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-description">'
        "Semantic retrieval searches the research paper for the most "
        "relevant evidence supporting or challenging the selected claim."
        "</div>",
        unsafe_allow_html=True,
    )

    claim_options = st.session_state["claim_list"]

    if claim_options:
        selected = st.selectbox(
            "Select a claim to verify",
            options=claim_options,
        )
    else:
        selected = st.text_area(
            "Enter one claim to verify",
            placeholder="Extract claims in Step 1, or paste a claim here.",
            height=110,
        )

    st.session_state["selected_claim"] = selected
    claim = st.session_state["selected_claim"]

    if st.button(
        "🔎 Retrieve Relevant Evidence",
        use_container_width=True,
    ) and claim:

        with st.spinner("RAG Retrieval Agent is searching the paper..."):
            try:
                results = vectorstore.similarity_search(
                    claim,
                    k=5,
                )
                st.session_state["evidence"] = results
                st.success("✅ Relevant evidence retrieved.")

            except Exception as exc:
                st.error(f"❌ Evidence retrieval failed: {exc}")


    if st.session_state["evidence"]:
        st.markdown("### 📚 Retrieved Evidence")

        for i, result in enumerate(
            st.session_state["evidence"],
            start=1,
        ):
            with st.expander(
                f"Evidence {i}",
                expanded=(i == 1),
            ):
                st.write(result.page_content)


    # ========================================================
    # STEP 3 — AI CLAIM VERIFICATION AGENT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="step-label">STEP 3</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-title">🤖 AI Claim Verification Agent</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-description">'
        "The verification agent reasons over the selected claim and "
        "retrieved evidence without introducing outside evidence."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "⚡ Verify Selected Claim",
        use_container_width=True,
    ) and claim:

        results = st.session_state["evidence"]

        if not results:
            with st.spinner("Retrieving evidence..."):
                try:
                    results = vectorstore.similarity_search(
                        claim,
                        k=5,
                    )
                    st.session_state["evidence"] = results
                except Exception as exc:
                    st.error(f"❌ Evidence retrieval failed: {exc}")
                    st.stop()

        evidence = "\n\n".join(
            f"EVIDENCE {i}:\n{result.page_content}"
            for i, result in enumerate(results, start=1)
        )

        verification_prompt = f"""
You are an AI research verification agent.

Evaluate whether the CLAIM is supported by the retrieved evidence
from the research paper.

CLAIM:
{claim}

RETRIEVED EVIDENCE:
{evidence}

Classify the claim as exactly one:
SUPPORTED
QUESTIONABLE
CONTRADICTED

Definitions:

SUPPORTED:
The retrieved evidence clearly supports the claim.

QUESTIONABLE:
The evidence is incomplete, indirect, ambiguous, or insufficient.

CONTRADICTED:
The retrieved evidence directly conflicts with the claim.

IMPORTANT:
- Do not invent evidence.
- Use ONLY the retrieved evidence.
- Pay special attention to exaggerated statements, unsupported
  conclusions, numerical claims, comparison claims, and words such
  as "all", "always", "best", "significant", "guarantees", and "proves".

Return exactly this structure:

CLASSIFICATION:
SUPPORTED / QUESTIONABLE / CONTRADICTED

CONFIDENCE:
0-100%

PROBLEMATIC PART:
Quote the specific phrase from the claim that appears unsupported,
questionable, or contradicted. If fully supported, write NONE.

REASONING:
Explain why the evidence supports or fails to support the claim.

EVIDENCE:
Identify which retrieved evidence is relevant.

LIMITATIONS:
Explain what cannot be concluded from the available evidence.
"""

        with st.spinner(
            "Verification Agent is analyzing the evidence..."
        ):
            try:
                verification_response = llm.invoke(
                    verification_prompt
                )
                st.session_state["verification_result"] = (
                    verification_response.content
                )
            except Exception as exc:
                st.error(f"❌ Verification failed: {exc}")


    # --------------------------------------------------------
    # VERIFICATION RESULT
    # --------------------------------------------------------

    if st.session_state["verification_result"]: 
 
        result_text = st.session_state["verification_result"] 
        result_upper = result_text.upper() 
 
        st.markdown("### 📊 AI Verification Result") 

        result = st.session_state["verification_result"].upper()

        if "UNSUPPORTED" in result:
            st.error("🔴 UNSUPPORTED")

        elif "CONTRADICTORY" in result:
            st.warning("🟡 CONTRADICTORY")

        elif "SUPPORTED" in result:
            st.success("🟢 SUPPORTED")

        else:
            st.write(st.session_state["verification_result"])

        st.write(result_text) 
 
        st.markdown("### 🎯 Claim Being Verified") 
        st.info(claim) 
 
        st.markdown("### 📚 Evidence Used") 
 
        for i, result in enumerate( 
            st.session_state["evidence"], 
            start=1, 
        ): 
            with st.expander(f"Evidence {i}"): 
                st.write(result.page_content)

    # ========================================================
    # FINAL STAGE — OVERALL PAPER ANALYSIS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="step-label">FINAL STAGE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-title">📊 Overall Research Paper Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-description">'
        "Generate an evidence-aware high-level assessment of the "
        "extracted claims and potential verification risks."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "📋 Generate Overall AI Report",
        use_container_width=True,
    ):

        claims = st.session_state["claims"]

        if not claims:
            st.warning("Please extract claims first.")

        else:
            report_prompt = f"""
You are a research paper assessment agent.

Based on the extracted claims below, create a concise,
evidence-aware overall assessment.

EXTRACTED CLAIMS:
{claims}

Provide:

OVERALL ASSESSMENT:

KEY RISKS:

CLAIMS THAT NEED FURTHER VERIFICATION:

IMPORTANT LIMITATIONS:

Do not claim that the entire research paper is scientifically
proven or disproven.

The system only evaluates claims based on available evidence.
"""

            with st.spinner(
                "AI Assessment Agent is generating the overall report..."
            ):
                try:
                    report_response = llm.invoke(report_prompt)
                    st.session_state["overall_report"] = (
                        report_response.content
                    )
                except Exception as exc:
                    st.error(f"❌ Overall report generation failed: {exc}")


    if st.session_state["overall_report"]:
        st.markdown("### 📄 Overall AI Assessment")
        st.write(st.session_state["overall_report"])


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#6b7280; padding:1rem;">
        <strong>AI-Powered Research Paper Verification</strong><br>
        GenAI • RAG • Agentic AI • Evidence-Aware Analysis
    </div>
    """,
    unsafe_allow_html=True,
)
