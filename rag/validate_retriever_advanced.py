#!/usr/bin/env python3
"""Comprehensive validation suite for advanced retriever quality.

Tests all 8 claim types to ensure consistent 0.80-0.95 confidence scores.
"""

import json
import logging
from pathlib import Path

from rag.retriever import LegalRetriever, RetrieverConfig


LOGGER = logging.getLogger(__name__)


# Test queries representing all 8 claim types
TEST_QUERIES: dict[str, str] = {
    "unpaid_wages": "My employer has not paid my salary for 3 months",
    "domestic_violence": "My spouse beats me and threatens me daily at home",
    "property_dispute": "I lost ownership of my inherited land to a neighbor",
    "wrongful_termination": "My company fired me without any legal reason or notice",
    "police_harassment": "Police arrested me without proper FIR or charges",
    "tenant_rights": "My landlord is illegally evicting me without proper notice",
    "consumer_fraud": "The seller gave me fake product and refuses to refund",
    "other": "I need legal help but not sure what type of case it is",
}


def validate_retriever(force_rebuild: bool = False) -> dict[str, dict]:
    """Run comprehensive validation across all claim types."""
    config = RetrieverConfig()
    retriever = LegalRetriever(config=config)
    
    LOGGER.info("Loading/building index (force_rebuild=%s)...", force_rebuild)
    retriever.load_or_build_index(
        force_rebuild=force_rebuild,
        max_documents=5000,  # Larger index for better coverage
    )
    
    results: dict[str, dict] = {}
    
    print("\n" + "=" * 100)
    print("ADVANCED RETRIEVER VALIDATION - All Claim Types")
    print("=" * 100)
    
    for claim_type, query in TEST_QUERIES.items():
        print(f"\n[{claim_type.upper()}]")
        print(f"Query: {query}")
        print("-" * 100)
        
        passages = retriever.query(query, top_k=5)
        
        scores = [p["score"] for p in passages]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Check quality metrics
        all_above_threshold = all(s >= 0.80 for s in scores)
        min_threshold_met = min_score >= 0.75  # Allow slight margin on minimum
        
        results[claim_type] = {
            "query": query,
            "scores": scores,
            "min_score": min_score,
            "max_score": max_score,
            "avg_score": avg_score,
            "all_above_0.80": all_above_threshold,
            "quality_status": "✅ PASS" if all_above_threshold else "⚠️  MARGINAL" if min_threshold_met else "❌ FAIL",
        }
        
        # Print scores
        for idx, passage in enumerate(passages, start=1):
            score = passage["score"]
            status = "✅" if score >= 0.80 else "⚠️ " if score >= 0.75 else "❌"
            preview = passage["passage"][:150].replace("\n", " ")
            print(f"{status} [{idx}] score={score:.4f}")
            print(f"    {preview}...\n")
        
        # Summary line
        print(f"Average Score: {avg_score:.4f} | Status: {results[claim_type]['quality_status']}")
    
    # Overall summary
    print("\n" + "=" * 100)
    print("SUMMARY REPORT")
    print("=" * 100)
    
    passed = sum(1 for r in results.values() if r["all_above_0.80"])
    marginal = sum(1 for r in results.values() if not r["all_above_0.80"] and r["min_score"] >= 0.75)
    failed = sum(1 for r in results.values() if r["min_score"] < 0.75)
    
    print(f"\nResults:")
    print(f"  ✅ Passed (all≥0.80):    {passed}/8")
    print(f"  ⚠️  Marginal (min≥0.75): {marginal}/8")
    print(f"  ❌ Failed (min<0.75):    {failed}/8")
    
    print(f"\nDetailed Results by Claim Type:")
    for claim_type, result in results.items():
        status = result["quality_status"]
        avg = result["avg_score"]
        min_s = result["min_score"]
        print(f"  {claim_type:20s} | {status} | Avg={avg:.4f} | Min={min_s:.4f}")
    
    # Overall verdict
    if passed == 8:
        verdict = "🎯 EXCELLENT - All claim types scoring 0.80+. Ready for production."
    elif passed + marginal == 8:
        verdict = "✅ GOOD - All claim types scoring 0.75+. Minor tuning may help."
    elif passed + marginal >= 6:
        verdict = "⚠️  ACCEPTABLE - Most claim types good. Some areas need improvement."
    else:
        verdict = "❌ NEEDS IMPROVEMENT - Significant scores below threshold."
    
    print(f"\nOVERALL VERDICT: {verdict}")
    
    # Save results
    results_file = Path("retriever_validation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Validate advanced retriever quality")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild index from datasets",
    )
    args = parser.parse_args()
    
    validate_retriever(force_rebuild=args.force_rebuild)
