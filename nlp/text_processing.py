"""Shared NLP preprocessing utilities for normalization, lemmatization, and legal NER."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


_IRREGULAR_LEMMAS: dict[str, str] = {
    "men": "man",
    "women": "woman",
    "children": "child",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "geese": "goose",
    "was": "be",
    "were": "be",
    "is": "be",
    "are": "be",
    "am": "be",
    "has": "have",
    "had": "have",
    "does": "do",
    "did": "do",
    "went": "go",
    "gone": "go",
}

_LOCATION_TERMS: tuple[str, ...] = (
    "maharashtra",
    "mumbai",
    "pune",
    "nagpur",
    "thane",
    "nashik",
    "india",
)


def normalize_text(text: str) -> str:
    """Normalize text for stable downstream matching."""
    normalized = (text or "").strip().lower()
    normalized = normalized.replace("\u2019", "'")
    normalized = re.sub(r"[^a-z0-9\s\-\/\.:,]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _simple_lemma(token: str) -> str:
    if token in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[token]
    if len(token) <= 3:
        return token

    # Conservative suffix rules to keep legal terms stable.
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if token.endswith("ed") and len(token) > 4:
        base = token[:-2]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def lemmatize_text(text: str) -> str:
    """Apply lightweight lemmatization to normalized text."""
    normalized = normalize_text(text)
    if not normalized:
        return ""
    tokens = re.findall(r"\b[a-z0-9]+\b", normalized)
    lemmas = [_simple_lemma(token) for token in tokens]
    return " ".join(lemmas)


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def extract_named_entities(text: str) -> dict[str, list[str]]:
    """Extract lightweight legal entities using regex/rule patterns.

    This is a practical NER layer for legal triage, not a full statistical NER model.
    """
    source = text or ""
    lowered = source.lower()
    entities: dict[str, list[str]] = defaultdict(list)

    # Date-like expressions.
    date_patterns = (
        r"\b\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}\b",
        r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}\b",
        r"\b(today|tomorrow|yesterday|tonight)\b",
    )
    for pattern in date_patterns:
        for match in re.findall(pattern, lowered):
            value = match if isinstance(match, str) else " ".join(match)
            entities["date"].append(value)

    # Monetary mentions.
    for match in re.findall(r"\b(?:rs\.?|inr|rupees?)\s*\d[\d,]*(?:\.\d+)?\b", lowered):
        entities["money"].append(match)
    for match in re.findall(r"\b\d[\d,]*(?:\.\d+)?\s*(?:rs\.?|inr|rupees?)\b", lowered):
        entities["money"].append(match)

    # Phone/helpline numbers.
    for match in re.findall(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b", lowered):
        entities["phone"].append(match)
    for match in re.findall(r"\b(1091|1098|112|181|15100)\b", lowered):
        entities["phone"].append(match)

    # FIR/case identifiers.
    for match in re.findall(r"\bfir\s*(?:no\.?|number)?\s*[\w\/-]+\b", lowered):
        entities["fir_reference"].append(match)

    # Statute/section entities.
    section_patterns = (
        r"\bsection\s*\d+[a-z]?\b",
        r"\bu/s\s*\d+[a-z]?\b",
        r"\b(?:ipc|crpc|bnss?|pwdva|posco|pocso)\b",
    )
    for pattern in section_patterns:
        for match in re.findall(pattern, lowered):
            entities["legal_reference"].append(match)

    # Courts/authorities.
    court_terms = (
        "supreme court",
        "high court",
        "district court",
        "family court",
        "consumer court",
        "labour court",
        "magistrate",
        "police station",
        "commissioner",
        "sp office",
    )
    for term in court_terms:
        if term in lowered:
            entities["authority"].append(term)

    # Locations (rule-list based).
    for term in _LOCATION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            entities["location"].append(term)

    return {key: _dedupe_preserve(value) for key, value in entities.items() if value}


def prepare_text_features(text: str) -> dict[str, Any]:
    """Return reusable normalization, lemmatization, and NER features."""
    normalized = normalize_text(text)
    lemmatized = lemmatize_text(normalized)
    return {
        "raw_text": text or "",
        "normalized_text": normalized,
        "lemmatized_text": lemmatized,
        "named_entities": extract_named_entities(text or ""),
    }
