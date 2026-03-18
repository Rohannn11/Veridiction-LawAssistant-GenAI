"""Knowledge loader for India/Maharashtra legal mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LegalKnowledgeBase:
    """Loads curated legal knowledge used for structured response grounding."""

    def __init__(self, knowledge_path: str | Path | None = None) -> None:
        self.knowledge_path = Path(knowledge_path or "data/legal_knowledge/maharashtra_legal_knowledge.json")
        self._data = self._load_data()

    def _load_data(self) -> dict[str, Any]:
        if not self.knowledge_path.exists():
            return {"claim_type_mapping": {}, "national_helplines": []}
        return json.loads(self.knowledge_path.read_text(encoding="utf-8"))

    @property
    def state(self) -> str:
        return str(self._data.get("state", "Maharashtra"))

    @property
    def country(self) -> str:
        return str(self._data.get("country", "India"))

    @property
    def national_helplines(self) -> list[dict[str, Any]]:
        return list(self._data.get("national_helplines", []))

    def claim_mapping(self, claim_type: str) -> dict[str, Any]:
        mapping = self._data.get("claim_type_mapping", {})
        claim = mapping.get(claim_type)
        if claim:
            return dict(claim)
        return dict(mapping.get("other", {}))
