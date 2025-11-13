"""Utility functions for interacting with the OpenAI API and
producing structured interview insights."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


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

    def _call(self, prompt: str) -> str:
        """Send a simple user prompt via the Chat Completions API."""

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return (completion.choices[0].message.content or "").strip()

    def extract_requirements(self, job_description: str) -> List[str]:
        prompt = (
            "You are an assistant that reads job descriptions and lists the 3-6 most critical "
            "skills, responsibilities, or qualifications that cannot be missed. Return only "
            "a bullet list without commentary. Job description:\n" + job_description
        )
        return _parse_bullets(self._call(prompt))

    def extract_strengths_and_weaknesses(
        self, cv_text: str, job_requirements: Iterable[str]
    ) -> Tuple[List[str], List[str]]:
        requirements_blob = "\n".join(f"- {req}" for req in job_requirements)
        prompt = (
            "Given the resume details below and the job requirements, highlight the key strengths "
            "(skills or experiences that align with the job) and weaknesses (gaps, missing "
            "experience, or areas to improve). Provide two bullet lists titled Strengths and "
            "Weaknesses.\nJob requirements:\n"
            f"{requirements_blob or '- Not available'}\n\nResume:\n{cv_text}"
        )
        response = self._call(prompt)
        strengths = _extract_section(response, "strengths")
        weaknesses = _extract_section(response, "weaknesses")
        return strengths, weaknesses

    def compute_match_report(
        self, job_requirements: Iterable[str], strengths: Iterable[str], weaknesses: Iterable[str]
    ) -> MatchReport:
        prompt = (
            "Act as an interview coach. Based on the job requirements, strengths, and weaknesses, "
            "estimate a percentage match score (0-100) and describe the likelihood of getting the "
            "role (High, Medium, or Low) with one concise justification. Respond in JSON with the "
            "shape {\"match_score\": int, \"likelihood\": str, \"reasoning\": str}.\n"
            f"Job requirements: {list(job_requirements)}\n"
            f"Strengths: {list(strengths)}\n"
            f"Weaknesses: {list(weaknesses)}"
        )
        import json

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

    def chat_response(
        self,
        messages: List[Dict[str, str]],
        job_requirements: Iterable[str],
        strengths: Iterable[str],
        weaknesses: Iterable[str],
    ) -> str:
        context = (
            "You are a concise interview preparation assistant. Keep answers short, specific, "
            "and grounded in the candidate's experience. Avoid speculation."
        )
        profile_context = _profile_blob(job_requirements, strengths, weaknesses)
        conversation: List[str] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            speaker = "Candidate" if msg["role"] == "user" else "Coach"
            conversation.append(f"{speaker}: {msg['content']}")
        prompt = (
            f"{context}\n\nProfile context:\n{profile_context}\n\n"
            "Conversation so far:\n"
            + "\n".join(conversation)
            + "\nCoach:"
        )
        return self._call(prompt)


def _parse_bullets(text: str) -> List[str]:
    bullets: List[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*• ").strip()
        if cleaned:
            bullets.append(cleaned)
    return bullets or [text.strip()]


def _extract_section(text: str, section_name: str) -> List[str]:
    """Extract bullet lines that belong to a section name."""
    lower = text.lower()
    section_name = section_name.lower()
    start = lower.find(section_name)
    if start == -1:
        return _parse_bullets(text)
    remaining = text[start:].split("\n", 1)
    body = remaining[1] if len(remaining) > 1 else remaining[0]
    return _parse_bullets(body)


def _extract_json(text: str) -> str:
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return "{}"


def _profile_blob(
    job_requirements: Iterable[str],
    strengths: Iterable[str],
    weaknesses: Iterable[str],
) -> str:
    return (
        "Job requirements: "
        + (", ".join(job_requirements) or "Not provided")
        + "\nStrengths: "
        + (", ".join(strengths) or "Not available")
        + "\nWeaknesses: "
        + (", ".join(weaknesses) or "Not available")
    ).strip()
