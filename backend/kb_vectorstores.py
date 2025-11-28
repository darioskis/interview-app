"""Vector store utilities for the knowledge-base JSON files.

This module loads role ratio guidance and soft-skill questions from disk,
validates them with Pydantic models, and builds separate Chroma vector stores
for downstream retrieval in the interview-coach pipeline.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class RoleQuestionRatio(BaseModel):
    """Represents the technical/soft mix and guidance for a role."""

    role_id: str
    role_name: str
    category: Literal["Technical", "Hybrid", "Non-Technical"]
    technical_percentage: int
    soft_skill_percentage: int
    prompt_instruction: str

    @classmethod
    def from_dict(cls, data: dict) -> "RoleQuestionRatio":
        """Create a RoleQuestionRatio from a raw dict, with validation."""
        try:
            return cls(
                role_id=data["role_id"],
                role_name=data["role_name"],
                category=data["category"],
                technical_percentage=int(data["weights"]["technical_percentage"]),
                soft_skill_percentage=int(data["weights"]["soft_skill_percentage"]),
                prompt_instruction=data["prompt_instruction"],
            )
        except KeyError as exc:
            raise ValueError(f"Missing required field for RoleQuestionRatio: {exc}") from exc


class SoftSkillQuestion(BaseModel):
    """Represents a soft-skill interview question scoped by seniority."""

    id: int
    seniority_level: str
    soft_skill: str
    question: str

    @classmethod
    def from_dict(cls, data: dict) -> "SoftSkillQuestion":
        """Create a SoftSkillQuestion from a raw dict, with validation."""
        try:
            return cls(
                id=int(data["id"]),
                seniority_level=data["seniority_level"],
                soft_skill=data["soft_skill"],
                question=data["question"],
            )
        except KeyError as exc:
            raise ValueError(f"Missing required field for SoftSkillQuestion: {exc}") from exc


def load_question_type_ratios(path: str | Path) -> list[RoleQuestionRatio]:
    """Load the role question type ratios JSON file into model instances."""

    kb_path = Path(path)
    logger.info("Loading question type ratios from %s", kb_path)
    try:
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Ratios file not found: {kb_path}") from exc

    if not isinstance(raw, list):
        raise ValueError("Ratios JSON must be a list of objects")

    ratios: list[RoleQuestionRatio] = []
    for entry in raw:
        try:
            ratios.append(RoleQuestionRatio.from_dict(entry))
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid role ratio entry: {entry}") from exc
    logger.info("Loaded %d role ratios", len(ratios))
    return ratios


def load_seniority_soft_skill_questions(path: str | Path) -> list[SoftSkillQuestion]:
    """Load the soft-skill questions JSON file into model instances."""

    kb_path = Path(path)
    logger.info("Loading soft-skill questions from %s", kb_path)
    try:
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Soft-skill questions file not found: {kb_path}") from exc

    if not isinstance(raw, list):
        raise ValueError("Soft-skill questions JSON must be a list of objects")

    questions: list[SoftSkillQuestion] = []
    for entry in raw:
        try:
            questions.append(SoftSkillQuestion.from_dict(entry))
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid soft-skill question entry: {entry}") from exc
    logger.info("Loaded %d soft-skill questions", len(questions))
    return questions


def _reset_directory(persist_directory: str | Path) -> Path:
    """Remove an existing persist directory to rebuild a vector store."""

    directory = Path(persist_directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_role_ratio_vectorstore(
    ratios: list[RoleQuestionRatio],
    embeddings: Embeddings,
    persist_directory: str | Path = "chroma_role_ratios",
) -> Chroma:
    """Build (or rebuild) a Chroma vector store for role question type ratios."""

    persist_dir = _reset_directory(persist_directory)
    docs: list[Document] = []
    for ratio in ratios:
        text_to_embed = (
            f"{ratio.role_name} ({ratio.category}). "
            f"Technical: {ratio.technical_percentage}%, "
            f"Soft: {ratio.soft_skill_percentage}%. "
            f"{ratio.prompt_instruction}"
        )
        docs.append(
            Document(
                page_content=text_to_embed,
                metadata={
                    "role_id": ratio.role_id,
                    "role_name": ratio.role_name,
                    "category": ratio.category,
                    "technical_percentage": ratio.technical_percentage,
                    "soft_skill_percentage": ratio.soft_skill_percentage,
                    "prompt_instruction": ratio.prompt_instruction,
                },
            )
        )

    logger.info("Building role ratio vector store at %s", persist_dir)
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    store.persist()
    logger.info("Persisted %d role ratio entries", len(docs))
    return store


def load_role_ratio_vectorstore(
    embeddings: Embeddings,
    persist_directory: str | Path = "chroma_role_ratios",
) -> Chroma:
    """Load an existing Chroma vector store for role question type ratios."""

    logger.info("Loading role ratio vector store from %s", persist_directory)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def build_soft_skill_question_vectorstore(
    questions: list[SoftSkillQuestion],
    embeddings: Embeddings,
    persist_directory: str | Path = "chroma_soft_skill_questions",
) -> Chroma:
    """Build (or rebuild) a Chroma vector store for seniority soft-skill questions."""

    persist_dir = _reset_directory(persist_directory)
    docs: list[Document] = []
    for q in questions:
        docs.append(
            Document(
                page_content=q.question,
                metadata={
                    "id": q.id,
                    "seniority_level": q.seniority_level,
                    "soft_skill": q.soft_skill,
                },
            )
        )

    logger.info("Building soft-skill question vector store at %s", persist_dir)
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    store.persist()
    logger.info("Persisted %d soft-skill question entries", len(docs))
    return store


def load_soft_skill_question_vectorstore(
    embeddings: Embeddings,
    persist_directory: str | Path = "chroma_soft_skill_questions",
) -> Chroma:
    """Load an existing Chroma vector store for seniority soft-skill questions."""

    logger.info("Loading soft-skill question vector store from %s", persist_directory)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def create_default_embeddings(use_azure: bool | None = False) -> Embeddings:
    """Create an embeddings client, defaulting to OpenAIEmbeddings.

    Azure OpenAI can be enabled by setting use_azure=True and configuring the
    expected environment variables for Azure credentials.
    """

    if use_azure:
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings()
    return OpenAIEmbeddings()


def _default_kb_paths(base_dir: Path | None = None) -> tuple[Path, Path]:
    base = base_dir or Path(__file__).resolve().parent / "KB"
    return base / "question_type_ratios.json", base / "seniority_soft_skills_questions.json"


def _summarize_store(name: str, store: Chroma) -> None:
    try:
        stats = store._collection.count()
        logger.info("%s vector store contains %d items", name, stats)
    except Exception:  # pragma: no cover - best-effort logging
        logger.info("%s vector store build complete", name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embeddings = create_default_embeddings()
    ratios_path, soft_path = _default_kb_paths()

    ratios = load_question_type_ratios(ratios_path)
    soft_questions = load_seniority_soft_skill_questions(soft_path)

    role_store = build_role_ratio_vectorstore(ratios, embeddings)
    _summarize_store("Role ratios", role_store)

    soft_store = build_soft_skill_question_vectorstore(soft_questions, embeddings)
    _summarize_store("Soft-skill questions", soft_store)
