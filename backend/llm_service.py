"""Utility functions for interacting with the OpenAI API and
producing structured interview insights."""
from __future__ import annotations

"""Utility functions for interacting with the OpenAI API and knowledge-base-driven interview flows."""

import logging
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

from backend.config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_TOP_P,
    PROMPTS_DIR,
    PROMPT_FILES,
)

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_PROMPTS = {
    "extract_requirements": (
        "You are an assistant that reads job descriptions and lists the 3-6 most "
        "critical skills, responsibilities, or qualifications that cannot be missed. "
        "Return only a bullet list without commentary.\nJob description:\n{job_description}"
    ),
    "strengths_weaknesses": (
        "Given the resume details below and the job requirements, highlight the key "
        "strengths (skills or experiences that align with the job) and weaknesses "
        "(gaps, missing experience, or areas to improve). Provide two bullet lists "
        "titled Strengths and Weaknesses.\nJob requirements:\n{requirements}\n\nResume:\n{cv_text}"
    ),
    "match_report": (
        "Act as an interview coach. Based on the job requirements, strengths, and "
        "weaknesses, estimate a percentage match score (0-100) and describe the "
        "likelihood of getting the role (High, Medium, or Low) with one concise "
        "justification. Respond in JSON with the shape {{\"match_score\": int, "
        "\"likelihood\": str, \"reasoning\": str}}.\nJob requirements: "
        "{job_requirements}\nStrengths: {strengths}\nWeaknesses: {weaknesses}"
    ),
    "chat_system": (
        "You are a concise interview preparation assistant. Keep answers short, specific, "
        "and grounded in the candidate's experience. Avoid speculation."
    ),
    "chat_prompt": (
        "Profile context:\n{profile_context}\n\nConversation so far:\n{conversation}\nCoach:"
    ),
    "chat_evaluator": (
        "You are a senior interviewer evaluating candidate answers."
        "You provide a quick and brief feedback on how well candidate answered question"
        "You return score out of 10 and general feedback on good things and what to improve in an answer"
        "This evaluation should be given after completion answering each question i.e. including follow up questions"
    ),
    "job_analysis": (
        "Extract the role title, seniority level, role type, and a concise list of 5-10 key requirements."
        "Seniority must be one of: Regular Employee, Team Lead, Manager, Director."
        "Role type must be either Technical or Non-technical."
        "Respond with JSON fields role_title, seniority, role_type, requirements."
        "\nFollow these format rules:\n{format_instructions}"
    ),
    "soft_question_selector": (
        "Given the seniority, key requirements, and candidate soft-skill questions, pick up to {max_questions} soft-skill"
        " interview questions that best match the requirements. Return JSON with selected_questions."
        "\nSeniority: {seniority}\nRequirements: {requirements}\nCandidates:\n{candidate_questions}\n{format_instructions}"
    ),
}


@dataclass
class MatchReport:
    """Structured information describing the job alignment."""

    match_score: int
    likelihood: str
    reasoning: str


@dataclass
class AnswerEvaluation:
    score: int
    summary: str
    improvements: List[str]


@dataclass
class JobAnalysis:
    role_title: str
    seniority: str
    role_type: str
    requirements: List[str]


@dataclass
class QuestionPlan:
    role_name: str
    technical_percentage: int
    soft_skill_percentage: int
    prompt_instruction: str
    category: str = ""


@dataclass
class SoftSkillQuestion:
    """Structured representation of a KB soft-skill interview question."""

    id: str
    soft_skill: str
    question: str

class LLMService:
    """Simple wrapper around the OpenAI Responses API."""

    def __init__(self, temperature: float = 0.8, top_p: float = 0.7, model: str = "gpt-4.1") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Create a .env file with the key before running the app."
            )
        self.temperature = OPENAI_TEMPERATURE if temperature is None else float(temperature)
        self.model = model or OPENAI_MODEL
        self.top_p = OPENAI_TOP_P if top_p is None else float(top_p)
        self.client = OpenAI(api_key=api_key)
        self._prompts = _load_prompts()
        self._langchain_llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            openai_api_key=api_key,
        )
        self._soft_skill_kb, self._question_ratio_kb = _load_kb()

    def _call(self, prompt: str, system: str | None = None) -> str:
        """Send a simple prompt via the Responses API with an optional system message."""
        try:
            messages = []
            if system:
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system}],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            )
            response = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                input=messages,
            )
            return _response_to_text(response)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("LLM prompt failed: %s", exc)
            raise RuntimeError("LLM call failed") from exc

    def analyze_job_post(self, job_description: str) -> JobAnalysis | None:
        """Use LangChain to extract role metadata and requirements from a JD."""

        class JobAnalysisSchema(BaseModel):
            role_title: str = Field(description="Role or position title")
            seniority: Literal[
                "Regular Employee",
                "Team Lead",
                "Manager",
                "Director",
            ] = Field(
                description="Seniority level using one of: Regular Employee, Team Lead, Manager, Director"
            )
            role_type: Literal["Technical", "Non-technical"] = Field(
                description="Position type: Technical or Non-technical"
            )
            requirements: List[str] = Field(description="List of key requirements")

        parser = JsonOutputParser(pydantic_object=JobAnalysisSchema)
        prompt_template = self._prompts["job_analysis"]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt_template,
                ),
                ("user", "{job_description}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self._langchain_llm | parser
        try:
            result: JobAnalysisSchema = chain.invoke({"job_description": job_description})
            requirements = result.requirements or []
            normalized_seniority = _normalize_seniority(result.seniority)
            normalized_role_type = _normalize_role_type(result.role_type, result.role_title)
            role_title_clean = result.role_title.strip()

            if role_title_clean:
                return JobAnalysis(
                    role_title=role_title_clean,
                    seniority=normalized_seniority,
                    role_type=normalized_role_type,
                    requirements=[r.strip() for r in requirements if r.strip()],
                )

            logger.warning("Job analysis returned an empty role title; falling back to heuristics")
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Job analysis failed: %s", exc)

        return _fallback_job_analysis(job_description, self)

    def extract_requirements(self, job_description: str) -> List[str]:
        prompt = self._prompts["extract_requirements"].format(
            job_description=job_description
        )
        try:
            return _parse_bullets(self._call(prompt))
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to extract requirements: %s", exc)
            return [f"Error extracting requirements: {exc}"]

    def extract_strengths_and_weaknesses(
        self, cv_text: str, job_requirements: Iterable[str]
    ) -> Tuple[List[str], List[str]]:
        requirements_blob = "\n".join(f"- {req}" for req in job_requirements)
        prompt = self._prompts["strengths_weaknesses"].format(
            requirements=requirements_blob or "- Not available", cv_text=cv_text
        )
        try:
            response = self._call(prompt)
            strengths = _extract_section(response, "strengths")
            weaknesses = _extract_section(response, "weaknesses")
            return strengths, weaknesses
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to extract strengths/weaknesses: %s", exc)
            return [f"Error extracting strengths: {exc}"], [f"Error extracting weaknesses: {exc}"]

    def question_plan(self, role_title: str, role_type: str) -> QuestionPlan:
        """Map the role to a technical/soft-skill ratio and prompt guidance."""

        normalized_role_type = _normalize_role_type(role_type, role_title)
        is_technical = normalized_role_type.lower().startswith("tech")
        default_plan = QuestionPlan(
            role_name=role_title or "Unknown role",
            technical_percentage=70 if is_technical else 40,
            soft_skill_percentage=30 if is_technical else 60,
            prompt_instruction=(
                "Balance technical depth with soft skills to assess collaboration and communication."
                if is_technical
                else "Emphasize soft-skill and leadership depth; keep technical questions light or scenario-based."
            ),
            category=normalized_role_type or "General",
        )

        best_score = 0.0
        chosen = None
        query = f"{role_title} {role_type}".lower()
        for entry in self._question_ratio_kb:
            name = str(entry.get("role_name", "")).lower()
            category = str(entry.get("category", "")).lower()
            score = max(
                SequenceMatcher(None, query, name).ratio(),
                SequenceMatcher(None, query, category).ratio(),
            )
            if name and name in query:
                score += 0.2
            if score > best_score:
                best_score = score
                chosen = entry

        if not chosen:
            return default_plan

        weights = chosen.get("weights", {})
        return QuestionPlan(
            role_name=chosen.get("role_name", default_plan.role_name),
            technical_percentage=int(weights.get("technical_percentage", default_plan.technical_percentage)),
            soft_skill_percentage=int(weights.get("soft_skill_percentage", default_plan.soft_skill_percentage)),
            prompt_instruction=str(chosen.get("prompt_instruction", default_plan.prompt_instruction)),
            category=str(chosen.get("category", normalized_role_type or default_plan.category)),
        )

    def select_soft_skill_questions(
        self, seniority: str, requirements: Iterable[str], max_questions: int = 3
    ) -> List[SoftSkillQuestion]:
        """Choose soft-skill questions from the KB using LangChain guidance.

        The flow follows two filters in order:
        1) Filter by seniority (strict match to the four allowed levels).
        2) Rank the remaining questions by similarity between their soft_skill label
           and the extracted requirements (e.g., surfacing "Time Management" when
           time management appears in requirements).
        Returns rich question metadata (ID, soft skill, text) so the chat layer can
        cite the originating KB item.
        """

        normalized_seniority = _normalize_seniority(seniority)
        seniority_lower = (normalized_seniority or "").lower()
        candidates = [
            item
            for item in self._soft_skill_kb
            if item.get("seniority_level", "").lower() == seniority_lower or not seniority_lower
        ]
        if not candidates:
            candidates = self._soft_skill_kb

        requirement_list = [r.strip() for r in requirements if str(r).strip()]
        scored_candidates = [
            (
                _soft_skill_requirement_score(str(item.get("soft_skill", "")), requirement_list),
                item,
            )
            for item in candidates
        ]
        scored_candidates.sort(key=lambda pair: pair[0], reverse=True)
        ranked_candidates = [item for _, item in scored_candidates]

        candidate_text = "\n".join(
            f"- ID-{c.get('id', '')} | {c.get('soft_skill')}: {c.get('question')}" for c in ranked_candidates
        )

        class Selection(BaseModel):
            selected_questions: List[str] = Field(description="List of chosen soft-skill interview questions")

        parser = JsonOutputParser(pydantic_object=Selection)
        prompt_template = self._prompts["soft_question_selector"]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt_template,
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self._langchain_llm | parser
        try:
            parsed: Selection = chain.invoke(
                {
                    "seniority": normalized_seniority or "Unknown",
                    "requirements": ", ".join(requirement_list) or "Not provided",
                    "candidate_questions": candidate_text,
                    "max_questions": max_questions,
                }
            )
            selected = [q.strip() for q in parsed.selected_questions if q.strip()]
            if selected:
                mapped: List[SoftSkillQuestion] = []
                for text in selected:
                    match = next(
                        (
                            SoftSkillQuestion(
                                id=str(c.get("id", "")).strip(),
                                soft_skill=str(c.get("soft_skill", "")).strip(),
                                question=str(c.get("question", "")).strip(),
                            )
                            for c in ranked_candidates
                            if str(c.get("question", "")).strip() == text
                        ),
                        None,
                    )
                    if match:
                        mapped.append(match)
                if mapped:
                    return mapped[:max_questions]
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Soft-skill selection failed: %s", exc)

        # Fallback to the top N ranked candidates if parsing failed
        fallback: List[SoftSkillQuestion] = []
        for c in ranked_candidates[:max_questions]:
            question_text = str(c.get("question", "")).strip()
            if not question_text:
                continue
            fallback.append(
                SoftSkillQuestion(
                    id=str(c.get("id", "")).strip(),
                    soft_skill=str(c.get("soft_skill", "")).strip(),
                    question=question_text,
                )
            )
        return fallback

    def compute_match_report(
        self, job_requirements: Iterable[str], strengths: Iterable[str], weaknesses: Iterable[str]
    ) -> MatchReport:
        prompt = self._prompts["match_report"].format(
            job_requirements=list(job_requirements),
            strengths=list(strengths),
            weaknesses=list(weaknesses),
        )
        import json

        try:
            raw = self._call(prompt)
            try:
                payload = json.loads(_extract_json(raw))
            except json.JSONDecodeError:
                # Provide a safe fallback if the model returns unexpected format
                return MatchReport(match_score=50, likelihood="Unknown", reasoning=raw[:200])
            return MatchReport(
                match_score=int(payload.get("match_score", 0)),
                likelihood=str(payload.get("likelihood", "Unknown")),
                reasoning=str(payload.get("reasoning", "")),
            )
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to compute match report: %s", exc)
            return MatchReport(match_score=0, likelihood="Unknown", reasoning=str(exc))

    def chat_response(
        self,
        messages: List[Dict[str, str]],
        job_requirements: Iterable[str],
        strengths: Iterable[str],
        weaknesses: Iterable[str],
        question_plan: QuestionPlan | None = None,
        soft_skill_questions: Iterable[SoftSkillQuestion] | None = None,
        job_description: str | None = None,
        cv_text: str | None = None,
        role_title: str | None = None,
        seniority: str | None = None,
        role_type: str | None = None,
    ) -> str:
        system_prompt_raw = self._prompts["chat_system"].strip()
        system_prompt = system_prompt_raw.format(
            job_description=job_description or "Not provided",
            cv_text=cv_text or "Not provided",
        )
        if question_plan:
            system_prompt += (
                "\n\nInterview mix guidance: focus ~{tech}% technical and ~{soft}% soft-skill questions. "
                "{instruction}"
            ).format(
                tech=question_plan.technical_percentage,
                soft=question_plan.soft_skill_percentage,
                instruction=question_plan.prompt_instruction,
            )
        if soft_skill_questions:
            formatted_soft = []
            for q in soft_skill_questions:
                soft_id = getattr(q, "id", "")
                soft_skill = getattr(q, "soft_skill", "")
                question_text = getattr(q, "question", "")
                if not question_text:
                    continue
                label = f"ID-{soft_id}" if soft_id else "KB"
                formatted_soft.append(f"• {label} | {soft_skill}: {question_text}")
            if formatted_soft:
                system_prompt += "\nPrioritize these soft-skill prompts when relevant:\n" + "\n".join(
                    formatted_soft
                )
        system_prompt += (
            "\nFor every new primary interview question you ask (not clarifying follow-ups), append a question "
            "type tag at the end: [Technical] for hard-skill topics or [Non-technical] for behavioral/soft-skill topics."
        )
        if soft_skill_questions:
            system_prompt += (
                " When you use one of the listed soft-skill prompts as a primary question, tag it as [Non-technical] and "
                "also add ' (Based on question ID-xx)' replacing xx with the provided ID. Do not add tags to follow-up "
                "clarifiers."
            )
        profile_context = _profile_blob(
            job_requirements,
            strengths,
            weaknesses,
            role_title=role_title,
            seniority=seniority,
            role_type=role_type,
        )
        conversation: List[str] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            speaker = "Candidate" if msg["role"] == "user" else "Coach"
            conversation.append(f"{speaker}: {msg['content']}")
        prompt = self._prompts["chat_prompt"].format(
            profile_context=profile_context, 
            conversation="\n".join(conversation)
        )
        try:
            return self._call(prompt, system=system_prompt)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to generate chat response: %s", exc)
            return "Sorry, I couldn't generate a reply right now."
            
    def evaluate_answer(self, question: str, answer: str) -> AnswerEvaluation | None:
        """Evaluate a candidate's answer to a given question.

        Returns an AnswerEvaluation or None if evaluation fails.
        """
        prompt = self._prompts["chat_evaluator"].format(
            question=question or "Not provided",
            answer=answer or "",
        )
        try:
            raw = self._call(prompt)
        except Exception as exc:
            logger.exception("Answer evaluation LLM call failed: %s", exc)
            return None

        try:
            data = _maybe_json(raw)
            score = int(data.get("score", 0))
            summary = str(data.get("summary", "")).strip()
            improvements = [str(x).strip() for x in data.get("improvements", []) if str(x).strip()]
            return AnswerEvaluation(score=score, summary=summary, improvements=improvements)
        except Exception as exc:
            logger.exception("Failed to parse answer evaluation JSON: %s; raw=%r", exc, raw)
            return None


def _response_to_text(response) -> str:
    """Extract the assistant text from a Responses API payload."""
    try:
        texts: List[str] = []
        output = getattr(response, "output", None) or []
        for item in output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    texts.append(text)
        if not texts:
            texts.extend(getattr(response, "output_text", []) or [])
        return "\n".join(texts).strip()
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to parse response text: %s", exc)
        return ""


def _parse_bullets(text: str) -> List[str]:
    try:
        bullets: List[str] = []
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-*• ").strip()
            if cleaned:
                bullets.append(cleaned)
        return bullets or [text.strip()]
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to parse bullets: %s", exc)
        return [text.strip()]


def _extract_section(text: str, section_name: str) -> List[str]:
    """Extract bullet lines that belong to a section name."""
    try:
        lower = text.lower()
        section_name = section_name.lower()
        other_section = "weaknesses" if section_name == "strengths" else "strengths"

        start = lower.find(section_name)
        if start == -1:
            return _parse_bullets(text)

        # Capture only the requested section by stopping at the next section header.
        end = lower.find(other_section, start + len(section_name))
        scoped = text[start:end] if end != -1 else text[start:]

        # Drop the heading line so only bullet content remains.
        lines = scoped.splitlines()
        if lines and section_name in lines[0].lower():
            lines = lines[1:]
        return _parse_bullets("\n".join(lines))
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to extract section '%s': %s", section_name, exc)
        return _parse_bullets(text)


def _extract_json(text: str) -> str:
    import re

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return "{}"
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to extract JSON: %s", exc)
        return "{}"


def _maybe_json(text: str) -> Dict:
    import json

    try:
        return json.loads(_extract_json(text))
    except Exception:
        return {}


def _soft_skill_requirement_score(soft_skill: str, requirements: Iterable[str]) -> float:
    """Score how well a soft skill aligns to the requirement list (case-insensitive).

    Seniority is handled before this scoring function; here we focus purely on
    similarity between the soft-skill label and the extracted requirements so that
    items like "Time Management" surface when that requirement appears or is implied.
    """

    label = (soft_skill or "").strip().lower()
    if not label:
        return 0.0

    requirement_list = [str(r).strip().lower() for r in requirements if str(r).strip()]
    if not requirement_list:
        return 0.0

    best = 0.0
    for req in requirement_list:
        if not req:
            continue
        if label in req or req in label:
            return 1.0
        best = max(best, SequenceMatcher(None, label, req).ratio())
    return best


def _profile_blob(
    job_requirements: Iterable[str],
    strengths: Iterable[str],
    weaknesses: Iterable[str],
    role_title: str | None = None,
    seniority: str | None = None,
    role_type: str | None = None,
) -> str:
    try:
        parts = []
        if role_title or seniority or role_type:
            parts.append(
                "Role: "
                + ", ".join(
                    filter(
                        None,
                        [
                            (role_title or "").strip(),
                            (seniority or "").strip(),
                            (role_type or "").strip(),
                        ],
                    )
                ).strip()
            )
        parts.append("Job requirements: " + (", ".join(job_requirements) or "Not provided"))
        parts.append("Strengths: " + (", ".join(strengths) or "Not available"))
        parts.append("Weaknesses: " + (", ".join(weaknesses) or "Not available"))
        return "\n".join(parts).strip()
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to build profile blob: %s", exc)
        return "Job requirements: Not provided\nStrengths: Not available\nWeaknesses: Not available"

def _load_prompts() -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for key, default in DEFAULT_PROMPTS.items():
        filename = PROMPT_FILES.get(key, f"{key}.txt")
        path = PROMPTS_DIR / filename
        prompts[key] = _read_prompt_file(path, default)
    return prompts


def _read_prompt_file(path: Path, default: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Prompt file %s missing; using default", path)
        return default
    except Exception as exc:
        logger.exception("Failed to read prompt file %s: %s", path, exc)
        return default


def _load_kb() -> Tuple[List[Dict], List[Dict]]:
    import json

    base = Path(__file__).resolve().parent / "KB"
    soft_skill_path = base / "seniority_soft_skills_questions.json"
    ratio_path = base / "question_type_ratios.json"

    def _safe_read_json(path: Path) -> List[Dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("KB file %s missing; using empty list", path)
            return []
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to load KB file %s: %s", path, exc)
            return []

    return _safe_read_json(soft_skill_path), _safe_read_json(ratio_path)


def _normalize_seniority(raw: str | None) -> str:
    """Map any free-form seniority string into one of four allowed values."""

    text = (raw or "").strip().lower()
    if not text:
        return "Regular Employee"

    if any(keyword in text for keyword in ["director", "head", "vp", "vice president"]):
        return "Director"
    if "manager" in text or "mgr" in text:
        return "Manager"
    if any(keyword in text for keyword in ["lead", "leader", "principal", "staff"]):
        return "Team Lead"

    return "Regular Employee"


def _normalize_role_type(raw: str | None, role_title: str | None = None) -> str:
    """Classify the position type as Technical or Non-technical."""

    merged = f"{raw or ''} {role_title or ''}".lower()
    if "non-technical" in merged:
        return "Non-technical"

    technical_keywords = [
        "engineer",
        "developer",
        "devops",
        "sre",
        "architect",
        "data",
        "analytics",
        "ml",
        "ai",
        "it ",
        "software",
        "systems",
        "security",
        "qa",
        "test",
    ]
    if any(keyword in merged for keyword in technical_keywords):
        return "Technical"

    non_technical_keywords = [
        "hr",
        "recruit",
        "talent",
        "sales",
        "marketing",
        "customer",
        "support",
        "success",
        "operations",
        "finance",
        "legal",
        "people",
    ]
    if any(keyword in merged for keyword in non_technical_keywords):
        return "Non-technical"

    return "Technical"


def _fallback_job_analysis(job_description: str, service: "LLMService") -> JobAnalysis:
    """Heuristic extraction when structured parsing fails or omits role details."""

    role_title = _guess_role_title(job_description)
    seniority = _normalize_seniority(role_title or job_description)
    role_type = _normalize_role_type(None, f"{role_title} {job_description}")

    requirements: List[str] = []
    try:
        requirements = service.extract_requirements(job_description)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Fallback requirement extraction failed: %s", exc)

    return JobAnalysis(
        role_title=role_title or "Role not detected",
        seniority=seniority,
        role_type=role_type,
        requirements=requirements,
    )


def _guess_role_title(job_description: str) -> str:
    """Pull the most plausible role title from the first descriptive lines."""

    try:
        lines = [line.strip(" -*\t") for line in job_description.splitlines() if line.strip()]
        for line in lines:
            lower = line.lower()
            if any(lower.startswith(prefix) for prefix in ["job title", "role", "position"]):
                candidate = line.split(":", 1)[-1].strip()
                if candidate:
                    return candidate
            if 1 <= len(line.split()) <= 12:
                return line
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Role title guess failed: %s", exc)

    return ""
