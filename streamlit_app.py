import logging
import time

import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

MAX_CALLS_PER_MINUTE = 20

from backend.llm_service import LLMService, MatchReport, SoftSkillQuestion

load_dotenv()


def validate_text(name: str, text: str, max_len: int = 100) -> str | None:
    text = text or ""
    if len(text) > max_len:
        return f"{name} is too long ({len(text)} characters). Please shorten it."
    return None


def allow_api_call() -> bool:
    now = time.time()
    st.session_state.api_calls = [
        ts for ts in st.session_state.api_calls if now - ts < 60
    ]
    if len(st.session_state.api_calls) >= MAX_CALLS_PER_MINUTE:
        return False
    st.session_state.api_calls.append(now)
    return True

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
        left: 35%;
        right: 3%;
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
if "job_role_title" not in st.session_state:
    st.session_state.job_role_title = ""
if "job_seniority" not in st.session_state:
    st.session_state.job_seniority = ""
if "job_role_type" not in st.session_state:
    st.session_state.job_role_type = ""
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
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = None
if "api_calls" not in st.session_state:
    st.session_state.api_calls = []

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
        jd_error = validate_text("Job description", st.session_state.job_description)
        cv_error = validate_text("CV / Resume", st.session_state.cv_text, max_len=10000)
        if jd_error or cv_error:
            if jd_error:
                st.error(jd_error)
            if cv_error:
                st.error(cv_error)
            st.stop()

        jd = st.session_state.job_description or ""
        cv = st.session_state.cv_text or ""
        has_jd = bool(jd.strip())
        has_cv = bool(cv.strip())

        with st.spinner("Running LLM tools..."):
            st.session_state.question_plan = None
            st.session_state.soft_skill_questions = []

            # 1) TOOL: Job profile
            if not allow_api_call():
                st.warning(
                    "You’ve reached the current AI usage limit (per minute). "
                    "Please wait a bit before trying again."
                )
                st.stop()
            if has_jd:
                job_profile = llm.run_job_profile_tool(jd)
            else:
                job_profile = llm.run_job_profile_tool("")
            st.session_state.job_role_title = job_profile.role_title
            st.session_state.job_seniority = job_profile.seniority
            st.session_state.job_role_type = job_profile.role_type
            st.session_state.job_requirements = job_profile.requirements

            # keep legacy keys for downstream compatibility
            st.session_state.role_title = st.session_state.job_role_title
            st.session_state.seniority_level = st.session_state.job_seniority
            st.session_state.role_type = st.session_state.job_role_type

            # 2) TOOL: Strengths & weaknesses
            if not allow_api_call():
                st.warning(
                    "You’ve reached the current AI usage limit (per minute). "
                    "Please wait a bit before trying again."
                )
                st.stop()
            if has_cv:
                sw = llm.run_strengths_weaknesses_tool(
                    cv,
                    st.session_state.job_requirements,
                )
            else:
                sw = llm.run_strengths_weaknesses_tool(
                    "",
                    st.session_state.job_requirements,
                )
            st.session_state.strengths = sw.strengths
            st.session_state.weaknesses = sw.weaknesses

            # 3) TOOL: Match report
            if not allow_api_call():
                st.warning(
                    "You’ve reached the current AI usage limit (per minute). "
                    "Please wait a bit before trying again."
                )
                st.stop()
            st.session_state.match_report = llm.compute_match_report(
                st.session_state.job_requirements,
                st.session_state.strengths,
                st.session_state.weaknesses,
            )

            # Question mix and soft-skill focus (uses outputs from tools above)
            if st.session_state.job_requirements:
                st.session_state.question_plan = llm.question_plan(
                    st.session_state.job_role_title, st.session_state.job_role_type
                )
                st.session_state.soft_skill_questions = llm.select_soft_skill_questions(
                    st.session_state.job_seniority, st.session_state.job_requirements
                )

        if not has_jd:
            st.warning(
                "You did not provide a job description. The role profile and match "
                "report are generic and may be less accurate."
            )
        if not has_cv:
            st.warning(
                "You did not provide a CV. Strengths and weaknesses are empty, and "
                "the match report is based only on the job side."
            )

    st.markdown("---")
    st.markdown("### Job Profile (Tool 1: Job analysis)")
    if (
        st.session_state.job_role_title
        or st.session_state.job_seniority
        or st.session_state.job_role_type
    ):
        st.write(
            f"{st.session_state.job_role_title or 'Role TBD'}"
            f" | {st.session_state.job_seniority or 'Seniority TBD'}"
            f" | {st.session_state.job_role_type or 'Role type TBD'}"
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
        st.write(
            "\n".join(
                f"• ID-{getattr(q, 'id', '')}: {getattr(q, 'question', '')}"
                f" ({getattr(q, 'soft_skill', '')})".rstrip()
                for q in st.session_state.soft_skill_questions
            )
        )
    else:
        st.info("Soft-skill prompts will show after role detection.")

    st.markdown("---")
    st.markdown("**Key job requirements**")
    if st.session_state.job_requirements:
        st.write("\n".join(f"• {req}\n" for req in st.session_state.job_requirements))
    else:
        st.info("Provide the job description to extract requirements.")

    st.markdown("### Strengths & Weaknesses (Tool 2: CV fit)")
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
    st.markdown("### Match Report (Tool 3: Match analysis)")
    st.metric("Match score", f"{st.session_state.match_report.match_score}%")
    st.write(f"Likelihood: {st.session_state.match_report.likelihood}")
    if st.session_state.match_report.reasoning:
        st.caption(st.session_state.match_report.reasoning)

with col_chat:
    st.subheader("Interview chat")
    if llm is None:
        st.stop()

    total_messages = len(st.session_state.messages)
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if (
                message["role"] == "assistant"
                and st.session_state.last_evaluation
                and idx == total_messages - 1
            ):
                ev = st.session_state.last_evaluation
                st.markdown("#### Answer evaluation (Tool 4: Answer scorer)")
                st.markdown(f"**Score:** {ev.score}/10")
                st.markdown(f"**Summary:** {ev.summary}")
                if ev.improvements:
                    st.markdown("**How to improve:**")
                    for tip in ev.improvements:
                        st.markdown(f"- {tip}")

    # Add breathing room so the floating input doesn't cover the last reply
    st.markdown("<div style='height: 80px'></div>", unsafe_allow_html=True)

    # Keep track of last assistant question (for evaluation context)
    if "last_assistant_message" not in st.session_state:
        st.session_state.last_assistant_message = ""

    if prompt := st.chat_input("Answer or ask something about your interview prep"):
        answer_error = validate_text("Your answer", prompt, max_len=4000)
        if answer_error:
            st.error(answer_error)
            st.stop()

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
            last_question = st.session_state.get("last_question", "")
            evaluation = None
            if last_question and st.session_state.job_requirements:
                if not allow_api_call():
                    st.warning(
                        "You’ve reached the current AI usage limit (per minute). "
                        "Please wait a bit before trying again."
                    )
                    st.stop()
                evaluation = llm.run_answer_evaluator_tool(
                    question=last_question,
                    answer=prompt,
                    job_requirements=st.session_state.job_requirements,
                )
                st.session_state.last_evaluation = evaluation
            else:
                st.session_state.last_evaluation = None

            # Get coach response
            with st.spinner("Thinking..."):
                if not allow_api_call():
                    st.warning(
                        "You’ve reached the current AI usage limit (per minute). "
                        "Please wait a bit before trying again."
                    )
                    st.stop()
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

            question_line = next(
                (
                    line[len("Question:") :].strip()
                    for line in response.splitlines()
                    if line.strip().lower().startswith("question:")
                ),
                "",
            )
            st.session_state.last_question = question_line

            # Render assistant response
            with st.chat_message("assistant"):
                st.markdown(response)

                if st.session_state.last_evaluation:
                    ev = st.session_state.last_evaluation
                    st.markdown("#### Answer evaluation (Tool 4: Answer scorer)")
                    st.markdown(f"**Score:** {ev.score}/10")
                    st.markdown(f"**Summary:** {ev.summary}")
                    if ev.improvements:
                        st.markdown("**How to improve:**")
                        for tip in ev.improvements:
                            st.markdown(f"- {tip}")
