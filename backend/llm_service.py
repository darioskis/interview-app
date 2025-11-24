"""Utility functions for interacting with the OpenAI API and
producing structured interview insights."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


PROMPT_DEFAULTS = {
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
}


@dataclass
class MatchReport:
    """Structured information describing the job alignment."""

    match_score: int
    likelihood: str
    reasoning: str


class LLMService:
    """Simple wrapper around the OpenAI Responses API."""

    def __init__(self, temperature: float = 0.1, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Create a .env file with the key before running the app."
            )
        self.temperature = temperature
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self._prompts = _load_prompts()

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
                input=messages,
            )
            return _response_to_text(response)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("LLM prompt failed: %s", exc)
            raise RuntimeError("LLM call failed") from exc

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
    ) -> str:
        system_prompt = self._prompts["chat_system"].strip()
        profile_context = _profile_blob(job_requirements, strengths, weaknesses)
        conversation: List[str] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            speaker = "Candidate" if msg["role"] == "user" else "Coach"
            conversation.append(f"{speaker}: {msg['content']}")
        prompt = self._prompts["chat_prompt"].format(
            profile_context=profile_context, conversation="\n".join(conversation)
        )
        try:
            return self._call(prompt, system=system_prompt)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Failed to generate chat response: %s", exc)
            return "Sorry, I couldn't generate a reply right now."


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


def _profile_blob(
    job_requirements: Iterable[str],
    strengths: Iterable[str],
    weaknesses: Iterable[str],
) -> str:
    try:
        return (
            "Job requirements: "
            + (", ".join(job_requirements) or "Not provided")
            + "\nStrengths: "
            + (", ".join(strengths) or "Not available")
            + "\nWeaknesses: "
            + (", ".join(weaknesses) or "Not available")
        ).strip()
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to build profile blob: %s", exc)
        return "Job requirements: Not provided\nStrengths: Not available\nWeaknesses: Not available"


def _load_prompts() -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for key, default in PROMPT_DEFAULTS.items():
        prompts[key] = _read_prompt_file(key, default)
    return prompts


def _read_prompt_file(name: str, default: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Prompt file %s missing; using default", path)
        return default
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to read prompt file %s: %s", path, exc)
        return default
