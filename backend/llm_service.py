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

    def __init__(self, temperature: float = 0.5, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Create a .env file with the key before running the app."
            )
        self.temperature = temperature
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def _call(self, prompt: str, system: str | None = None) -> str:
        """Send a simple prompt via the Responses API with an optional system message."""

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

    def extract_requirements(self, job_description: str) -> List[str]:
        prompt = (
            """You are a recruiting analyst. From a job description extract required:
            - hard skills
            - soft skills
            - experiences
            
            Rephrase extracted requirements in 2-3 words each. Be specific. List out top 7 that looks most important. 
            Check if these 7 are really the most important to the position. Add brief explanation why you think so.
            
            Return only a numbered list with each requirement starting in a new row on a webpage.

            
            Job description:\n""" + job_description
        )
        return _parse_bullets(self._call(prompt))

    def extract_strengths_and_weaknesses(
        self, cv_text: str, job_requirements: Iterable[str]
    ) -> Tuple[List[str], List[str]]:
        requirements_blob = "\n".join(f"- {req}" for req in job_requirements)
        prompt = (
            f"""Given the resume details below and the job requirements, highlight the key strengths
            (skills or experiences that align with the job) and weaknesses (gaps, missing
            experience, or areas to improve). Each category should have up to 5 highlighted items, named in 2-3 words each.
            
            Revise items in both lists and look for contradictory findings such as same or similar skill or experience is in both categories. \
            example. Strength - ERP/CRM Implementation experience, weakness - little ERP/CRM Implementation experience. 

            Format:
            Return two numbered lists titled Strengths and Weaknesses with corresponding items (identified strengths and weaknesses) under each of them. 
            Each extracted item should start in a new row on a website. 
            
            \nJob requirements:\n
            {requirements_blob or '- Not available'}\n\nResume:\n{cv_text}"""
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
        system_prompt = (
            """You are a job interview coach. You goal is to prepare user to crush job interview by preparing most likely \
            questions and how to answer them correctly. 
            
            First, prioritize a role play where you are hiring manager, and user is a candidate. \
            Identify up to 5 most important requirements for the position and ask 2-3 questions for each. \
            Provide questions one by one i.e. if user's answer requires follow-up question do it immediately. \
            When answers looks complete, move to the next question.
        
            After role play, as a recruitment expert, give feedback on what was good in the answers and how to improve in order to nail during an interview. 
            If user stops answering, do not create answers on behalf of him. But continue generate generating questions with possible answers from your expertise for the feedback.
            
            Train how to highlight strengths, and how to cover weaknesses, include examples.

            If user feels good about preparation and there's nothing more to help, generate a brief summary on key points \
            from the discussion as a quick reminder. 

            Do not respond if user uses any illegal or unetchical questions/answers ex. racist, xenophobic, gender related etc. 

            If candidate is overqualified a lot, do not do role play. Just tell to the user that one is much overqualified. 
            """
        )
        profile_context = _profile_blob(job_requirements, strengths, weaknesses)
        conversation: List[str] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            speaker = "Candidate" if msg["role"] == "user" else "Coach"
            conversation.append(f"{speaker}: {msg['content']}")
        prompt = (
            "Profile context:\n"
            f"{profile_context}\n\n"
            "Conversation so far:\n"
            + "\n".join(conversation)
            + "\nCoach:"
        )
        return self._call(prompt, system=system_prompt)


def _response_to_text(response) -> str:
    """Extract the assistant text from a Responses API payload."""

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
