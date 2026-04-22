import json
import os
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "vedantu_study_materials"


def _load_materials() -> list[dict[str, Any]]:
    with (DATA_DIR / "study_materials.json").open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("materials", [])


def _material_to_document(material: dict[str, Any]) -> Document:
    page_content = "\n".join(
        [
            f"Title: {material.get('title', '')}",
            f"Topic: {material.get('topic', '')}",
            f"Type: {material.get('type', '')}",
            f"Description: {material.get('description', '')}",
            f"URL: {material.get('url', '')}",
        ]
    )
    return Document(
        page_content=page_content,
        metadata={
            "material_id": material.get("material_id", ""),
            "topic": material.get("topic", ""),
            "title": material.get("title", ""),
            "type": material.get("type", ""),
            "url": material.get("url", ""),
        },
    )


def _db_has_index() -> bool:
    sqlite_file = CHROMA_DIR / "chroma.sqlite3"
    return CHROMA_DIR.exists() and sqlite_file.exists()


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Create the Chroma index once, then reuse the local persisted database."""
    if force_rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = get_embeddings()

    if _db_has_index():
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required before indexing study materials.")

    documents = [_material_to_document(material) for material in _load_materials()]
    if not documents:
        raise ValueError("No study materials found in data/study_materials.json.")

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def get_retriever(search_k: int = 4):
    vectorstore = build_or_load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": search_k})
