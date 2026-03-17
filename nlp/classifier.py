"""Step 2: English-first legal claim classifier for Veridiction.

Hybrid strategy:
1) Rule-based keyword scoring for determinism and speed.
2) Embedding similarity using all-MiniLM-L6-v2 for semantic fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


LOGGER = logging.getLogger(__name__)


class ClassifierError(RuntimeError):
    """Raised when claim classification fails."""


@dataclass(slots=True)
class ClassifierConfig:
    """Configuration for claim classification."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_claim_type: str = "other"
    min_confidence: float = 0.33
    embedding_weight: float = 0.65
    keyword_weight: float = 0.35


@dataclass(slots=True)
class ClaimResult:
    """Structured classifier output."""

    claim_type: str
    urgency: str
    confidence: float
    rationale_short: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "urgency": self.urgency,
            "confidence": round(self.confidence, 4),
            "rationale_short": self.rationale_short,
        }


class ClaimClassifier:
    """Classifies user legal issues into normalized claim categories."""

    CLAIM_LABELS: tuple[str, ...] = (
        "unpaid_wages",
        "domestic_violence",
        "property_dispute",
        "wrongful_termination",
        "police_harassment",
        "tenant_rights",
        "consumer_fraud",
        "other",
    )

    CLAIM_PROTOTYPES: dict[str, str] = {
        "unpaid_wages": "Employer has not paid salary wages overtime dues labor payment pending",
        "domestic_violence": "Domestic violence abuse by husband partner family physical threats harassment",
        "property_dispute": "Land property ownership boundary inheritance title conflict possession dispute",
        "wrongful_termination": "Job termination fired illegally without notice compensation labor rights",
        "police_harassment": "Police harassment unlawful detention threats abuse bribery FIR refusal",
        "tenant_rights": "Tenant eviction rent deposit landlord harassment lease rights",
        "consumer_fraud": "Consumer fraud cheating defective service refund denied scam",
        "other": "General legal issue not matching common categories",
    }

    KEYWORDS: dict[str, tuple[str, ...]] = {
        "unpaid_wages": (
            "unpaid",
            "salary",
            "wage",
            "wages",
            "overtime",
            "did not pay",
            "payment pending",
            "minimum wage",
            "labour",
            "labor",
        ),
        "domestic_violence": (
            "domestic violence",
            "husband beat",
            "physical abuse",
            "dowry",
            "threat at home",
            "marital abuse",
        ),
        "property_dispute": (
            "property",
            "land",
            "inheritance",
            "boundary",
            "encroachment",
            "title deed",
            "partition",
        ),
        "wrongful_termination": (
            "fired",
            "terminated",
            "dismissed",
            "job loss",
            "without notice",
            "illegal termination",
        ),
        "police_harassment": (
            "police",
            "fir not filed",
            "custody",
            "detained",
            "threatened by police",
            "bribe",
        ),
        "tenant_rights": (
            "landlord",
            "tenant",
            "eviction",
            "rent",
            "security deposit",
            "lease",
        ),
        "consumer_fraud": (
            "fraud",
            "scam",
            "refund denied",
            "defective product",
            "cheated",
            "consumer complaint",
        ),
    }

    HIGH_URGENCY_PATTERNS: tuple[str, ...] = (
        "urgent",
        "immediately",
        "threat",
        "violence",
        "beating",
        "assault",
        "in danger",
        "detained",
        "arrest",
        "tonight",
    )

    MEDIUM_URGENCY_PATTERNS: tuple[str, ...] = (
        "notice",
        "deadline",
        "hearing",
        "summons",
        "salary due",
        "eviction",
        "termination",
        "non payment",
    )

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("Loading classifier embeddings on device: %s", device)
        self._model = SentenceTransformer(self.config.model_name, device=device)
        self._prototype_embeddings = self._model.encode(
            [self.CLAIM_PROTOTYPES[label] for label in self.CLAIM_LABELS],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def classify(self, query: str) -> dict[str, Any]:
        """Classify a legal query into claim type and urgency."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty English sentence.")

        text = query.strip()
        try:
            keyword_scores = self._keyword_scores(text)
            embedding_scores = self._embedding_scores(text)
            combined_scores = self._combine_scores(keyword_scores, embedding_scores)

            best_label, best_score = max(combined_scores.items(), key=lambda item: item[1])
            if best_score < self.config.min_confidence:
                best_label = self.config.fallback_claim_type

            urgency = self._detect_urgency(text, best_label)
            rationale = self._build_rationale(best_label, keyword_scores, embedding_scores)

            result = ClaimResult(
                claim_type=best_label,
                urgency=urgency,
                confidence=float(best_score),
                rationale_short=rationale,
            )
            return result.to_dict()
        except Exception as exc:  # pragma: no cover
            raise ClassifierError(f"Classification failed: {exc}") from exc

    def _keyword_scores(self, text: str) -> dict[str, float]:
        lowered = text.lower()
        scores = {label: 0.0 for label in self.CLAIM_LABELS}

        for label, patterns in self.KEYWORDS.items():
            hits = 0
            for pattern in patterns:
                if re.search(rf"\b{re.escape(pattern)}\b", lowered):
                    hits += 1
            if patterns:
                scores[label] = hits / len(patterns)

        if all(value == 0.0 for value in scores.values()):
            scores["other"] = 0.1
        return scores

    def _embedding_scores(self, text: str) -> dict[str, float]:
        query_embedding = self._model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = cos_sim(query_embedding, self._prototype_embeddings).squeeze(0)

        scores: dict[str, float] = {}
        for idx, label in enumerate(self.CLAIM_LABELS):
            score = float(similarities[idx].item())
            scores[label] = max(0.0, min(1.0, (score + 1.0) / 2.0))
        return scores

    def _combine_scores(
        self,
        keyword_scores: dict[str, float],
        embedding_scores: dict[str, float],
    ) -> dict[str, float]:
        combined: dict[str, float] = {}
        for label in self.CLAIM_LABELS:
            combined[label] = (
                self.config.keyword_weight * keyword_scores.get(label, 0.0)
                + self.config.embedding_weight * embedding_scores.get(label, 0.0)
            )
        return combined

    def _detect_urgency(self, text: str, claim_type: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in self.HIGH_URGENCY_PATTERNS):
            return "high"

        if claim_type in {"domestic_violence", "police_harassment"}:
            return "high"

        if any(token in lowered for token in self.MEDIUM_URGENCY_PATTERNS):
            return "medium"

        if claim_type in {"unpaid_wages", "property_dispute", "wrongful_termination", "tenant_rights"}:
            return "medium"

        return "low"

    def _build_rationale(
        self,
        claim_type: str,
        keyword_scores: dict[str, float],
        embedding_scores: dict[str, float],
    ) -> str:
        return (
            f"Selected {claim_type} using hybrid match "
            f"(keyword={keyword_scores.get(claim_type, 0.0):.2f}, "
            f"embedding={embedding_scores.get(claim_type, 0.0):.2f})."
        )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify legal query claim type and urgency.")
    parser.add_argument("--query", type=str, default=None, help="Single query to classify")
    parser.add_argument(
        "--run-samples",
        action="store_true",
        help="Run built-in English sample queries",
    )
    return parser


def _sample_queries() -> list[str]:
    return [
        "My employer has not paid my salary for three months.",
        "My husband is threatening and physically hurting me at home.",
        "My landlord is forcing eviction without notice.",
        "Police refused to file my FIR and demanded bribe.",
        "I was fired without notice after reporting unsafe conditions.",
        "My cousin occupied our inherited agricultural land.",
        "A company took payment but never delivered the product.",
        "Need help understanding legal process for a contract dispute.",
        "I am in danger and being assaulted right now.",
        "My rent deposit is not being returned.",
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_cli().parse_args()
    classifier = ClaimClassifier()

    if args.run_samples:
        for sample in _sample_queries():
            result = classifier.classify(sample)
            print(json.dumps({"query": sample, "result": result}, ensure_ascii=True))
        return

    query = args.query or "My employer has not paid my wages for 2 months."
    result = classifier.classify(query)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
