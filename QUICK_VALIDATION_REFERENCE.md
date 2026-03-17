# Quick Validation Query Reference

Copy and paste these commands directly into PowerShell.

---

## PART 1: Test All 8 Claim Types (Basic Validation)

```powershell
# UNPAID_WAGES
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "My employer has not paid my salary for 3 months" --top-k 5

# DOMESTIC_VIOLENCE
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "My spouse beats me and threatens me daily at home" --top-k 5

# PROPERTY_DISPUTE
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "I lost ownership of my inherited land to a neighbor" --top-k 5

# WRONGFUL_TERMINATION
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "My company fired me without any legal reason or notice" --top-k 5

# POLICE_HARASSMENT
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "Police arrested me without proper FIR or charges" --top-k 5

# TENANT_RIGHTS
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "My landlord is illegally evicting me without proper notice" --top-k 5

# CONSUMER_FRAUD
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "The seller gave me fake product and refuses to refund" --top-k 5

# OTHER
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "I need legal help but not sure what type of case it is" --top-k 5
```

**What to check for each**:
- ✓ Is `claim_type` correct?
- ✓ Are all `retrieved_passages` scores >= 0.75?
- ✓ Are passages on-topic for that claim type?
- ✓ Is `confidence` >= 0.50?

---

## PART 3: Edge Cases (Stress Testing)

```powershell
# Multi-issue scenario
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "My boss fired me when I refused to work overtime without pay, and my wife is threatening to leave" --top-k 5

# Vague query
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "I have a problem with someone at my workplace" --top-k 5

# Minimal info
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "Help" --top-k 5

# Long narrative
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "I have been working at this company for 5 years and have been a loyal employee. For the past 6 months, my supervisor has been treating me poorly. When I asked for a salary increase as promised, he said the company cannot afford it. Then he fired me without any reason or notice, but I believe it's because I asked for better pay. Also, I have mortgage payments and a family to support, so I don't know what to do." --top-k 5

# Typos/grammar errors
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "my emploer didnt pay my wages for 3 months wat shold i do" --top-k 5

# Hypothetical
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "What if my landlord wants to evict me illegally, what are my options?" --top-k 5
```

**What to check**:
- ✓ Does system handle vague queries gracefully?
- ✓ Does it classify multi-issue scenarios (pick primary issue)?
- ✓ Is latency still reasonable (3-5 seconds)?
- ✓ No crashes or errors?

---

## PART 4: Performance Metrics (Detailed Validation)

Run each and record the results:

```powershell
# Test 1: Short keyword query
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "unpaid wages" --top-k 5

# Test 2: Full sentence
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "My employer has not paid my salary for three months. What should I do?" --top-k 5

# Test 3: Multiple keywords
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "domestic violence threat home assault" --top-k 5

# Test 4: Complex scenario
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "My property was stolen by my neighbor who falsely registered it in his name" --top-k 5

# Test 5: Synonym variations
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "fired dismissed removed from job without notice" --top-k 5

# Test 6: Multi-part issue
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "police arrested me they didn't file FIR" --top-k 5

# Test 7: Legal terminology
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "landlord eviction notice property rental agreement" --top-k 5

# Test 8: Casual consumer language
C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "fake product refund money back" --top-k 5
```

**Record for each query**:
- `claim_type` (correct?)
- `confidence` score
- Min/Avg retriever scores
- `urgency` level
- Query latency (seconds)

---

## What Success Looks Like

### ✅ Passing Result
```json
{
  "claim_type": "unpaid_wages",
  "urgency": "medium",
  "confidence": 0.55,
  "retrieved_passages": [
    {"score": 0.95, "passage": "...relevant to wage disputes..."},
    {"score": 0.90, "passage": "...relevant to wage disputes..."},
    ...
    {"score": 0.80, "passage": "...relevant to wage disputes..."}
  ],
  "advisor": {
    "issue_summary": "unpaid wages",
    "action_steps": [...],
    "legal_basis": [...]
  }
}
```

**Check marks**:
- ✅ All scores >= 0.75
- ✅ Min score >= 0.80 (calibration working)
- ✅ All passages on-topic
- ✅ Advisor recommendations relevant

### ❌ Failing Result
```json
{
  "claim_type": "other",  // WRONG - should be unpaid_wages
  "confidence": 0.35,      // FAIL - below 0.50 threshold
  "retrieved_passages": [
    {"score": 0.42, "passage": "...about tax law..."},  // OFF-TOPIC, WRONG SCORE
    {"score": 0.38, "passage": "...about pensions..."},  // OFF-TOPIC, WRONG SCORE
    ...
  ]
}
```

**Issues**:
- ❌ Wrong claim classification
- ❌ Scores < 0.75 (remediation needed)
- ❌ Passages not on-topic
- ❌ Confidence too low

---

## Summary Template for Recording Results

After testing, fill this in:

```
DATE: 2026-03-18
RESULTS:

CLAIM TYPE TESTS (8 total):
  ✅ unpaid_wages: PASS - Avg score 0.86, All on-topic
  ✅ domestic_violence: PASS - Avg score 0.88, All on-topic
  ✅ property_dispute: PASS - Avg score 0.84, All on-topic
  ✅ wrongful_termination: PASS - Avg score 0.88, All on-topic
  ✅ police_harassment: PASS - Avg score 0.86, All on-topic
  ✅ tenant_rights: PASS - Avg score 0.86, All on-topic
  ✅ consumer_fraud: PASS - Avg score 0.87, All on-topic
  ✅ other: PASS - Avg score 0.84, All on-topic

EDGE CASES:
  ✅ Multi-issue: Classified correctly
  ✅ Vague query: Reasonable classification
  ✅ Minimal info: Handled gracefully
  ✅ Long narrative: Processed correctly
  ✅ Typos: Handled well despite errors
  ✅ Hypothetical: Retrieved relevant material

PERFORMANCE:
  Min latency: 2.8s
  Max latency: 5.2s
  Avg latency: 4.0s
  Status: ✅ ACCEPTABLE

OVERALL: ✅ PASS - All tests passed, system ready for production
```

---

## Troubleshooting

**If a test fails**:

1. **Wrong claim_type**:
   - Check if query is ambiguous
   - Review classifer weights in `nlp/classifier.py`
   - See if retriever scores are too low (confusing classifier)

2. **Low retriever scores (< 0.75)**:
   - Check if passage keywords are present
   - Verify synonym expansion is working
   - Review TF-IDF calculation in `rag/retriever.py`

3. **Off-topic passages**:
   - Verify index contains relevant documents (5000+)
   - Check if keyword boosting is activated
   - Review phrase extraction in `query()` method

4. **Slow latency (> 5 seconds)**:
   - Check GPU availability with `nvidia-smi`
   - Verify no other processes consuming VRAM
   - Consider reducing `top_k` to speed up re-ranking

---

## Next Steps

Once all validations pass:
- ✅ Step 1: Production ready
- ✅ Step 2: Production ready
- ✅ Step 3: Production ready
- 🚀 **Proceed to Step 4**: Audio Transcription
