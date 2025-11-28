import streamlit as st
from dotenv import load_dotenv

from backend.llm_service import LLMService, MatchReport

load_dotenv()

st.set_page_config(page_title="Interview Coach", layout="wide")

# Keep the chat input visible and pinned to the bottom of the viewport, similar to ChatGPT's floating composer
st.markdown(
    """
    <style>
    /* Prevent content from being hidden behind the fixed composer */
    main .block-container {
        padding-bottom: 150px;
    }

    div[data-testid="stChatInput"] {
        position: fixed;
        left: 0;
        right: 0;
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
if "role_title" not in st.session_state:
    st.session_state.role_title = ""
if "seniority_level" not in st.session_state:
    st.session_state.seniority_level = ""
if "role_type" not in st.session_state:
    st.session_state.role_type = ""
if "job_requirements" not in st.session_state:
    st.session_state.job_requirements = []
if "strengths" not in st.session_state:
    st.session_state.strengths = []
if "weaknesses" not in st.session_state:
    st.session_state.weaknesses = []
if "soft_skill_questions" not in st.session_state:
    st.session_state.soft_skill_questions = []
if "match_report" not in st.session_state:
    st.session_state.match_report = MatchReport(match_score=0, likelihood="N/A", reasoning="")
if "question_plan" not in st.session_state:
    st.session_state.question_plan = None

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
            st.session_state.question_plan = None
            st.session_state.soft_skill_questions = []
            st.session_state.role_title = ""
            st.session_state.seniority_level = ""
            st.session_state.role_type = ""
            analysis = None
            if st.session_state.job_description:
                analysis = llm.analyze_job_post(st.session_state.job_description)
                if analysis:
                    st.session_state.role_title = analysis.role_title
                    st.session_state.seniority_level = analysis.seniority
                    st.session_state.role_type = analysis.role_type
                    st.session_state.job_requirements = analysis.requirements
                else:
                    st.session_state.job_requirements = llm.extract_requirements(
                        st.session_state.job_description
                    )
            else:
                st.session_state.job_requirements = []

            if st.session_state.job_requirements:
                st.session_state.question_plan = llm.question_plan(
                    st.session_state.role_title, st.session_state.role_type
                )
                st.session_state.soft_skill_questions = llm.select_soft_skill_questions(
                    st.session_state.seniority_level, st.session_state.job_requirements
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
    st.markdown("**Detected role**")
    if st.session_state.role_title or st.session_state.seniority_level or st.session_state.role_type:
        st.write(
            f"{st.session_state.role_title or 'Role TBD'}"
            f" | {st.session_state.seniority_level or 'Seniority TBD'}"
            f" | {st.session_state.role_type or 'Role type TBD'}"
        )
    else:
        st.info("Provide a job description to detect the role and seniority.")

    st.markdown("**Question mix guidance**")
    if st.session_state.question_plan:
        qp = st.session_state.question_plan
        st.write(
            f"Technical: {qp.technical_percentage}% | Soft skills: {qp.soft_skill_percentage}%"
        )
        st.caption(qp.prompt_instruction)
    else:
        st.info("Question ratio appears once a role is detected.")

    st.markdown("**Soft-skill focus**")
    if st.session_state.soft_skill_questions:
        st.write("\n".join(f"• {q}\n" for q in st.session_state.soft_skill_questions))
    else:
        st.info("Soft-skill prompts will show after role detection.")

    st.markdown("---")
    st.markdown("**Key job requirements**")
    if st.session_state.job_requirements:
        st.write("\n".join(f"• {req}\n" for req in st.session_state.job_requirements))
    else:
        st.info("Provide the job description to extract requirements.")

    st.markdown("**Your strengths**")
    if st.session_state.strengths:
        st.write("\n".join(f"• {item}\n" for item in st.session_state.strengths))
    else:
        st.info("Paste your CV/resume to identify strengths.")

    st.markdown("**Potential weaknesses**")
    if st.session_state.weaknesses:
        st.write("\n".join(f"• {item}\n" for item in st.session_state.weaknesses))
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

    # Keep track of last assistant question (for evaluation context)
    if "last_assistant_message" not in st.session_state:
        st.session_state.last_assistant_message = ""

    if prompt := st.chat_input("Answer or ask something about your interview prep"):
        # Simple security guard – don't call API for obviously inappropriate content
        banned_words = ["racist", "xenophobic", "illegal activity"]
        if any(word in prompt.lower() for word in banned_words):
            st.error("Your message contains inappropriate content. Please rephrase.")
        else:
            # Store user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Evaluate the answer based on last assistant message (often the question)
            last_question = st.session_state.last_assistant_message
            evaluation = None
            if last_question:
                evaluation = llm.evaluate_answer(last_question, prompt)

            # Get coach response
            with st.spinner("Thinking..."):
                response = llm.chat_response(
                    st.session_state.messages,
                    st.session_state.job_requirements,
                    st.session_state.strengths,
                    st.session_state.weaknesses,
                    question_plan=st.session_state.question_plan,
                    soft_skill_questions=st.session_state.soft_skill_questions,
                    job_description=st.session_state.job_description,
                    cv_text=st.session_state.cv_text,
                    role_title=st.session_state.role_title,
                    seniority=st.session_state.seniority_level,
                    role_type=st.session_state.role_type,
                )

            # Save assistant message (used as "last question" next time)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_assistant_message = response

            # Render assistant response
            with st.chat_message("assistant"):
                st.markdown(response)

                # If we managed to evaluate the user's answer, show it under the bot reply
                if evaluation:
                    st.markdown("---")
                    st.markdown(f"**Answer rating:** {evaluation.score} / 10")
                    if evaluation.summary:
                        st.markdown(f"**Summary:** {evaluation.summary}")
                    if evaluation.improvements:
                        st.markdown("**How to improve your answer:**")
                        for item in evaluation.improvements:
                            st.markdown(f"- {item}")
