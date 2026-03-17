#!/usr/bin/env python3
"""Validation Queries for Veridiction Step 1-3 Pipeline

Run individual queries manually to test retriever quality, classifier accuracy,
and advisor recommendation quality. Each query represents a real user scenario.

Usage:
  python agents/langgraph_flow.py --query "Your query here" --top-k 5
"""

# ==============================================================================
# PART 1: TEST QUERIES FOR EACH CLAIM TYPE (8 Scenarios)
# ==============================================================================

TEST_QUERIES_BY_CLAIM_TYPE = {
    "unpaid_wages": [
        "My employer has not paid my salary for 3 months",
        "My boss owes me 6 months of wages and refuses to pay",
        "I worked for a year but company never paid me anything",
        "Factory owner withheld my monthly compensation",
        "My salary has been pending for 2 months after I resigned",
    ],
    
    "domestic_violence": [
        "My spouse beats me and threatens me daily at home",
        "My husband assaults me physically and verbally",
        "I am being abused by my family members at home",
        "My wife threatens to harm me and my children",
        "My in-laws are violent and threatening towards me",
    ],
    
    "property_dispute": [
        "I lost ownership of my inherited land to a neighbor",
        "My ancestral property was taken by a relative unlawfully",
        "I have a property dispute with my co-owner over land",
        "Someone claims ownership of my property fraudulently",
        "My family member is preventing me from accessing my inherited house",
    ],
    
    "wrongful_termination": [
        "My company fired me without any legal reason or notice",
        "I was terminated without proper explanation or compensation",
        "My boss fired me for asking for better wages",
        "I was dismissed without following proper procedures",
        "The company removed me from my job illegally",
    ],
    
    "police_harassment": [
        "Police arrested me without proper FIR or charges",
        "Police refused to file my FIR when I complained",
        "I was beaten by police during arrest without reason",
        "Police extorted money from me unlawfully",
        "Police are demanding bribe to file my complaint",
    ],
    
    "tenant_rights": [
        "My landlord is illegally evicting me without proper notice",
        "Landlord increased rent unreasonably and wants me out",
        "My lease was terminated unlawfully by property owner",
        "Landlord took my deposit and is refusing to return it",
        "Owner is trying to evict me without legal procedure",
    ],
    
    "consumer_fraud": [
        "The seller gave me fake product and refuses to refund",
        "I bought defective goods and company won't replace them",
        "Online seller took my money but never delivered product",
        "Product quality is completely different from what was promised",
        "Shop owner sold me counterfeit items at premium price",
    ],
    
    "other": [
        "I need legal help but not sure what type of case it is",
        "I have a dispute but cannot categorize it properly",
        "Please help me understand my legal rights in general",
        "I want to know what laws apply to my situation",
        "Can you guide me through a legal matter",
    ],
}

# ==============================================================================
# PART 3: EDGE CASE & STRESS TEST QUERIES
# ==============================================================================

EDGE_CASE_QUERIES = [
    # Multi-issue scenarios
    "My boss fired me when I refused to work overtime without pay, and my wife is threatening to leave",
    "Police arrested me falsely, destroyed my property in their vehicle, and I lost my job as a result",
    "My landlord is evicting me without notice AND demanding money I never owed",
    
    # Vague/ambiguous queries
    "I have a problem with someone at my workplace",
    "There's a situation with family members",
    "I need help with a legal matter regarding property",
    
    # Minimal information
    "Help",
    "What can I do?",
    "Is this legal?",
    
    # Regional/Hindi terms mixed in
    "Mera baap ne meri property par claim kar diya (My father is claiming my property)",
    "Employers ne mere wages nahi diye (Employers didn't pay my wages)",
    
    # Very long narrative
    "I have been working at this company for 5 years and have been a loyal employee. For the past 6 months, my supervisor has been treating me poorly. When I asked for a salary increase as promised, he said the company cannot afford it. Then he fired me without any reason or notice, but I believe it's because I asked for better pay. Also, I have mortgage payments and a family to support, so I don't know what to do.",
    
    # Typos and grammatical errors
    "my emploer didnt pay my wages for 3 months wat shold i do",
    "someone has took my property without my permissin",
    
    # Hypothetical scenarios
    "What if my employer doesn't pay me for work I did?",
    "If my landlord wants to evict me illegally, what are my options?",
]

# ==============================================================================
# PART 4: PERFORMANCE VALIDATION QUERIES WITH EXPECTED METRICS
# ==============================================================================

PERFORMANCE_VALIDATION_QUERIES = [
    {
        "query": "unpaid wages",
        "claim_type": "unpaid_wages",
        "expected_retriever_score_min": 0.75,
        "expected_retriever_score_avg": 0.80,
        "expected_urgency": "medium",
        "expected_confidence_min": 0.50,
        "description": "One-word query - tests basic keyword matching",
    },
    {
        "query": "My employer has not paid my salary for three months. What should I do?",
        "claim_type": "unpaid_wages",
        "expected_retriever_score_min": 0.75,
        "expected_retriever_score_avg": 0.85,
        "expected_urgency": "medium",
        "expected_confidence_min": 0.55,
        "description": "Full sentence - tests phrase matching and context understanding",
    },
    {
        "query": "domestic violence threat home assault",
        "claim_type": "domestic_violence",
        "expected_retriever_score_min": 0.75,
        "expected_retriever_score_avg": 0.85,
        "expected_urgency": "high",
        "expected_confidence_min": 0.60,
        "description": "Multiple keywords - tests keyword weighting and prioritization",
    },
    {
        "query": "My property was stolen by my neighbor who falsely registered it in his name",
        "claim_type": "property_dispute",
        "expected_retriever_score_min": 0.75,
        "expected_retriever_score_avg": 0.82,
        "expected_urgency": "high",
        "expected_confidence_min": 0.50,
        "description": "Complex scenario - tests synonym expansion and multi-concept retrieval",
    },
    {
        "query": "fired dismissed removed from job without notice",
        "claim_type": "wrongful_termination",
        "expected_retriever_score_min": 0.75,
        "expected_retriever_score_avg": 0.85,
        "expected_urgency": "medium",
        "expected_confidence_min": 0.55,
        "description": "Synonym variations - tests synonym boosting",
    },
    {
        "query": "police arrested me they didn't file FIR",
        "claim_type": "police_harassment",
        "expected_retriever_score_min": 0.75,
        "expected_retriever_score_avg": 0.84,
        "expected_urgency": "high",
        "expected_confidence_min": 0.60,
        "description": "Multi-part issue - tests handling of compound scenarios",
    },
    {
        "query": "landlord eviction notice property rental agreement",
        "claim_type": "tenant_rights",
        "expected_retriever_score_min": 0.75,
        "expected_retriever_score_avg": 0.83,
        "expected_urgency": "high",
        "expected_confidence_min": 0.55,
        "description": "Legal terminology - tests domain-specific keyword recognition",
    },
    {
        "query": "fake product refund money back",
        "claim_type": "consumer_fraud",
        "expected_retriever_score_min": 0.75,
        "expected_retriever_score_avg": 0.84,
        "expected_urgency": "medium",
        "expected_confidence_min": 0.50,
        "description": "Consumer language - tests casual consumer query handling",
    },
]

# ==============================================================================
# MANUAL TESTING INSTRUCTIONS
# ==============================================================================

MANUAL_TESTING_GUIDE = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    MANUAL VALIDATION TESTING GUIDE                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

PART 1: TEST ALL CLAIM TYPES (8 Queries)
────────────────────────────────────────────

For each query below, run:
  python agents/langgraph_flow.py --query "YOUR_QUERY" --top-k 5

Then check:
  ✓ claim_type is correct (matches expected type)
  ✓ urgency is reasonable (low/medium/high)
  ✓ confidence >= 0.50
  ✓ Each retrieved_passage has score >= 0.75
  ✓ Passages are on-topic for the claim type

QUERIES:
─────────────────────────────────────

1. UNPAID_WAGES:
   "My employer has not paid my salary for 3 months"
   Expected: claim_type=unpaid_wages, urgency=medium, passages about wage disputes

2. DOMESTIC_VIOLENCE:
   "My spouse beats me and threatens me daily at home"
   Expected: claim_type=domestic_violence, urgency=high, passages about family violence

3. PROPERTY_DISPUTE:
   "I lost ownership of my inherited land to a neighbor"
   Expected: claim_type=property_dispute, urgency=high, passages about property ownership

4. WRONGFUL_TERMINATION:
   "My company fired me without any legal reason or notice"
   Expected: claim_type=wrongful_termination, urgency=medium, passages about termination law

5. POLICE_HARASSMENT:
   "Police arrested me without proper FIR or charges"
   Expected: claim_type=police_harassment, urgency=high, passages about arrest procedures

6. TENANT_RIGHTS:
   "My landlord is illegally evicting me without proper notice"
   Expected: claim_type=tenant_rights, urgency=high, passages about eviction law

7. CONSUMER_FRAUD:
   "The seller gave me fake product and refuses to refund"
   Expected: claim_type=consumer_fraud, urgency=medium, passages about consumer rights

8. OTHER:
   "I need legal help but not sure what type of case it is"
   Expected: claim_type=other, urgency=low, generic legal guidance


PART 3: EDGE CASES & STRESS TESTS (10 Queries)
────────────────────────────────────────────────

Run these to test robustness:

MULTI-ISSUE (System should pick primary issue):
  "My boss fired me when I refused unpaid overtime, and my wife is leaving me"
  → Should classify as wrongful_termination (primary) or domestic_violence

VAGUE QUERIES (System should still classify reasonably):
  "I have a problem with someone at my workplace"
  → Should classify as wrongful_termination or other

MINIMAL INFO (Stress test):
  "Help"
  "What can I do?"

LONG NARRATIVE (Real user behavior):
  "I have been working at this company for 5 years and have been a loyal 
   employee. For the past 6 months, my supervisor has been treating me poorly. 
   When I asked for a salary increase as promised, he said the company cannot 
   afford it. Then fired me without notice..."
  → Should classify as unpaid_wages + wrongful_termination (composite)

TYPOS/GRAMMAR (Real user patterns):
  "my emploer didnt pay my wages for 3 months wat shold i do"
  → Should still classify correctly despite errors

HYPOTHETICALS (Future-oriented):
  "What if my landlord wants to evict me illegally?"
  → Should still retrieve relevant material even in hypothetical form


PART 4: PERFORMANCE METRICS (Quantitative Validation)
───────────────────────────────────────────────────────

For each query below, create a CSV with results:

Query | Claim Type | Retriever Score (Min) | Retriever Score (Avg) | 
Classifier Confidence | Urgency | Advisor Quality (1-5)

Template:
  ---
  query: "unpaid wages"
  expected_claim_type: "unpaid_wages"
  actual_claim_type: "unpaid_wages" ✓
  retriever_scores: [0.95, 0.90, 0.88, 0.85, 0.80]
  retriever_min: 0.80 (Target: >= 0.75) ✓
  retriever_avg: 0.876 (Target: >= 0.80) ✓
  classifier_confidence: 0.558 (Target: >= 0.50) ✓
  urgency: "medium" (Reasonable for unpaid wages) ✓
  passage_relevance: All on-topic about wage disputes ✓
  ---

SAMPLE QUERIES TO QUANTIFY:

  1. "unpaid wages" (short, common)
  2. "My employer has not paid my salary for three months..." (full sentence)
  3. "domestic violence threat home assault" (multiple keywords)
  4. "My property was stolen by my neighbor..." (complex scenario)
  5. "fired dismissed removed from job without notice" (synonym variations)
  6. "police arrested me they didn't file FIR" (multi-part)
  7. "landlord eviction notice property rental agreement" (legal terminology)
  8. "fake product refund money back" (casual consumer language)


SUCCESS CRITERIA
─────────────────

✅ PASS if:
  • All 8 claim types: claim_type correct
  • All 8 claim types: retriever_scores >= 0.75
  • All 8 claim types: retriever_avg >= 0.80
  • All 8 claim types: passages on-topic
  • Edge cases: System doesn't crash, returns reasonable classification
  • Performance: Latency 3-5 seconds per query

❌ FAIL if:
  • Any claim type: Wrong classification
  • Any claim type: Retriever scores < 0.70
  • Any query: Passages off-topic
  • Any query: System errors or crashes
  • Performance: Latency > 10 seconds

REPORTING
──────────

After testing, create a validation report showing:
  1. Summary table of all 8 claim types (pass/fail/scores)
  2. Edge case results (pass/fail + observations)
  3. Performance metrics (latency distribution, score ranges)
  4. Any failures with specific examples
  5. Recommendations for improvements

Use JSON format for automated processing:

{
  "validation_date": "2026-03-18",
  "test_results": {
    "claim_types": {
      "unpaid_wages": {
        "query": "My employer has not paid my salary for 3 months",
        "result": "PASS",
        "claim_type_correct": true,
        "retriever_scores": [0.95, 0.90, 0.88, 0.85, 0.80],
        "retriever_min": 0.80,
        "retriever_avg": 0.876,
        "classifier_confidence": 0.558,
        "urgency": "medium",
        "passages_on_topic": true,
        "advisor_quality": 5
      },
      ...
    },
    "edge_cases": [...],
    "performance": {
      "avg_latency_ms": 4200,
      "min_latency_ms": 2800,
      "max_latency_ms": 5900
    }
  },
  "summary": {
    "claim_types_passed": 8,
    "claim_types_failed": 0,
    "overall_status": "PASS"
  }
}
"""

if __name__ == "__main__":
    print(MANUAL_TESTING_GUIDE)
    
    print("\n" + "=" * 80)
    print("PART 1: TEST QUERIES BY CLAIM TYPE")
    print("=" * 80)
    
    for claim_type, queries in TEST_QUERIES_BY_CLAIM_TYPE.items():
        print(f"\n[{claim_type.upper()}]")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
    
    print("\n" + "=" * 80)
    print("PART 3: EDGE CASE QUERIES")
    print("=" * 80)
    
    for i, query in enumerate(EDGE_CASE_QUERIES, 1):
        print(f"\n{i}. {query}")
    
    print("\n" + "=" * 80)
    print("PART 4: PERFORMANCE VALIDATION QUERIES")
    print("=" * 80)
    
    for metric in PERFORMANCE_VALIDATION_QUERIES:
        print(f"\nQuery: {metric['query']}")
        print(f"  Expected claim_type: {metric['claim_type']}")
        print(f"  Expected retriever min score: {metric['expected_retriever_score_min']}")
        print(f"  Expected retriever avg score: {metric['expected_retriever_score_avg']}")
        print(f"  Expected urgency: {metric['expected_urgency']}")
        print(f"  Expected confidence min: {metric['expected_confidence_min']}")
        print(f"  Description: {metric['description']}")
