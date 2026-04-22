import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from src.database import DATA_DIR, get_retriever


def _load_json(filename: str) -> dict:
    with (Path(DATA_DIR) / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_subject(subject: Optional[str]) -> Optional[str]:
    if not subject:
        return None
    normalized = subject.strip().lower()
    aliases = {
        "math": "mathematics",
        "maths": "mathematics",
        "science": "science",
    }
    return aliases.get(normalized, normalized)


@tool
def search_study_materials(query: str) -> str:
    """Search Vedantu study materials for a student's topic, weakness, study plan, or test-prep query."""
    print(f"Searching study materials for query: '{query}'")
    retriever = get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "No matching study materials were found."

    results = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        results.append(
            "\n".join(
                [
                    f"{index}. {metadata.get('title', 'Untitled')}",
                    f"   Topic: {metadata.get('topic', 'Unknown')}",
                    f"   Type: {metadata.get('type', 'Unknown')}",
                    f"   Link: {metadata.get('url', 'No URL available')}",
                    f"   Why relevant: {doc.page_content}",
                ]
            )
        )

    return "\n\n".join(results)


@tool
def get_upcoming_tests(subject: Optional[str] = None, days_ahead: int = 14) -> str:
    """Return upcoming tests for the default student, optionally filtered by subject and time window."""
    print(f"Fetching upcoming tests for subject: '{subject}' within the next {days_ahead} days")
    payload = _load_json("upcoming_tests.json")
    today = date.today()
    horizon = today + timedelta(days=days_ahead)
    subject_filter = _normalize_subject(subject)

    matching_tests = []
    for test in payload.get("upcoming_tests", []):
        test_date = datetime.strptime(test["date"], "%Y-%m-%d").date()
        if test_date < today or test_date > horizon:
            continue
        if subject_filter and subject_filter not in test["subject"].lower():
            continue
        matching_tests.append(test)

    if not matching_tests:
        return f"No upcoming tests found in the next {days_ahead} days."

    lines = []
    for test in sorted(matching_tests, key=lambda item: item["date"]):
        topics = ", ".join(test.get("topics", []))
        lines.append(
            f"{test['test_name']} ({test['subject']}) on {test['date']}; topics: {topics}"
        )
    return "\n".join(lines)


TOOLS = [search_study_materials, get_upcoming_tests]
