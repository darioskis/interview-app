# backend/config.py
from __future__ import annotations

import os
from pathlib import Path

# Base dir = project root (adjust if needed)
BASE_DIR = Path(__file__).resolve().parents[1]

# OpenAI / LLM settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.8"))
OPENAI_TOP_P = float(os.getenv("OPENAI_TOP_P", "0.7"))

# Embeddings for RAG
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

# Prompts
PROMPTS_DIR = BASE_DIR / "backend" / "prompts"

PROMPT_FILES = {
    "extract_requirements": "extract_requirements.txt",
    "strengths_weaknesses": "strengths_weaknesses.txt",
    "match_report": "match_report.txt",
    "chat_system": "chat_system.txt",
    "chat_prompt": "chat_prompt.txt",
    "chat_evaluator": "chat_evaluator.txt",
    "job_analysis": "job_analysis.txt",
    "soft_question_selector": "soft_question_selector.txt",
}
