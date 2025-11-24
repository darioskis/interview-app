import streamlit as st
from dotenv import load_dotenv

from backend.llm_service import LLMService, MatchReport

load_dotenv()

st.set_page_config(page_title="Interview Coach", layout="wide")

# Keep the chat input visible, similar to ChatGPT's floating composer
st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        z-index: 1000;
        background: linear-gradient(
            180deg,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.92) 35%,
            rgba(255, 255, 255, 1) 100%
        );
        border-top: 1px solid #e6e6e6;
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stChatInput"] > div {
        margin-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You help candidates prepare concise interview answers. Always ask clarifying "
                "questions when context is missing."
            ),
        }
    ]

if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""
if "job_requirements" not in st.session_state:
    st.session_state.job_requirements = []
if "strengths" not in st.session_state:
    st.session_state.strengths = []
if "weaknesses" not in st.session_state:
    st.session_state.weaknesses = []
if "match_report" not in st.session_state:
    st.session_state.match_report = MatchReport(match_score=0, likelihood="N/A", reasoning="")

llm = None
try:
    llm = LLMService()
except RuntimeError as exc:
    st.error(str(exc))

st.title("Your Job Interview Preparation Coach")
st.markdown(
    "This is AI based job interview preparation coach which would help you crush your next interview. For the best experience and result paste the **job description** and your **CV/resume**. The coach will extract key requirements, "
    "identify your strengths and weaknesses, and help you prepare for your interview."
)

col_overview, col_chat = st.columns([1, 2])

with col_overview:
    st.subheader("Profile Overview")
    with st.form("profile_form"):
        st.session_state.job_description = st.text_area(
            "Job description", value=st.session_state.job_description, height=180
        )
        st.session_state.cv_text = st.text_area(
            "CV / Resume", value=st.session_state.cv_text, height=180,
            help="Paste the relevant parts of your resume or LinkedIn profile."
        )
        submitted = st.form_submit_button("Analyze profile")

    if submitted and llm:
        with st.spinner("Extracting requirements and strengths..."):
            if st.session_state.job_description:
                st.session_state.job_requirements = llm.extract_requirements(
                    st.session_state.job_description
                )
            if st.session_state.cv_text:
                strengths, weaknesses = llm.extract_strengths_and_weaknesses(
                    st.session_state.cv_text, st.session_state.job_requirements
                )
                st.session_state.strengths = strengths
                st.session_state.weaknesses = weaknesses
            if st.session_state.job_requirements and st.session_state.strengths:
                st.session_state.match_report = llm.compute_match_report(
                    st.session_state.job_requirements,
                    st.session_state.strengths,
                    st.session_state.weaknesses,
                )

    st.markdown("---")
    st.markdown("**Key job requirements**")
    if st.session_state.job_requirements:
        st.write("\n".join(f"• {req}" for req in st.session_state.job_requirements))
    else:
        st.info("Provide the job description to extract requirements.")

    st.markdown("**Your strengths**")
    if st.session_state.strengths:
        st.write("\n".join(f"• {item}" for item in st.session_state.strengths))
    else:
        st.info("Paste your CV/resume to identify strengths.")

    st.markdown("**Potential weaknesses**")
    if st.session_state.weaknesses:
        st.write("\n".join(f"• {item}" for item in st.session_state.weaknesses))
    else:
        st.info("Weaknesses appear here once your CV is analyzed.")

    st.markdown("---")
    st.markdown("**Match assessment**")
    st.metric("Match score", f"{st.session_state.match_report.match_score}%")
    st.write(f"Likelihood: {st.session_state.match_report.likelihood}")
    if st.session_state.match_report.reasoning:
        st.caption(st.session_state.match_report.reasoning)

with col_chat:
    st.subheader("Interview chat")
    if llm is None:
        st.stop()

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add breathing room so the floating input doesn't cover the last reply
    st.markdown("<div style='height: 80px'></div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Ask for coaching or practice answering a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking..."):
            response = llm.chat_response(
                st.session_state.messages,
                st.session_state.job_requirements,
                st.session_state.strengths,
                st.session_state.weaknesses,
                st.session_state.job_description,
                st.session_state.cv_text
            )
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
