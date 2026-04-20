# NLP Concepts Used in Veridiction (and Why They Are Used)

## Scope

This document is only about NLP/GenAI concepts used in this project and the reason each concept is included.

It covers concepts implemented across:
- nlp/classifier.py
- nlp/text_processing.py
- rag/retriever.py
- agents/langgraph_flow.py
- audio/transcriber.py
- tts/speak.py

---

## 1. Speech-to-Text (ASR) for Voice Intake

## Concept
Automatic Speech Recognition (ASR) converts user audio to text so downstream NLP can operate on a unified textual input.

## Where it is used
- audio/transcriber.py

## How it is implemented
- faster-whisper model (distil-large-v3) is used to transcribe audio segments.
- Beam search decoding is used (beam_size default 5).
- VAD filtering (voice activity detection) is used to reduce non-speech noise.
- Segment outputs are concatenated into a final transcript.

## Why it is used
- Legal users may prefer speaking over typing.
- A single text representation allows the same classifier/retriever/advisor pipeline for both text and audio users.
- VAD and beam search improve transcript reliability in noisy conditions.

---

## 2. Hybrid Text Classification (Rules + Embeddings)

## Concept
Hybrid classification combines symbolic NLP (rules/keywords) with semantic NLP (dense embeddings).

## Where it is used
- nlp/classifier.py

## How it is implemented
1. Rule-based keyword scoring per claim type.
2. Sentence embedding similarity to claim prototypes using all-MiniLM-L6-v2.
3. Weighted fusion:

$$
\text{score}_{label} = 0.35 \cdot \text{keyword}_{label} + 0.65 \cdot \text{embedding}_{label}
$$

4. Threshold fallback to other when confidence is low.

## Why it is used
- Rules provide deterministic behavior and explainability.
- Embeddings provide semantic robustness to paraphrases and varied user wording.
- Weighted fusion reduces brittle behavior from either pure rules or pure semantics.

---

## 3. Text Normalization and Lemmatization

## Concept
Before scoring/retrieval routing, user text is normalized and lemmatized to reduce surface-form variance.

## Where it is used
- nlp/text_processing.py
- nlp/classifier.py
- rag/retriever.py

## Why it is used
- Improves robustness against inflectional and formatting variation (for example wages/wage, fired/firing).
- Stabilizes keyword and intent matching.

---

## 4. Lightweight Legal Named Entity Recognition (NER)

## Concept
Rule-based legal NER extracts high-value entities from user queries.

## Where it is used
- nlp/text_processing.py
- nlp/classifier.py (entities included in classification payload)
- rag/retriever.py (entities persisted in retrieval metadata)
- agents/langgraph_flow.py (entities exposed in final response and claim profile)

## Entity categories extracted
- date
- money
- phone
- fir_reference
- legal_reference
- authority
- location

## Why it is used
- Captures critical legal facts for explainability and triage quality.
- Supports downstream follow-up logic and output transparency.

---

## 5. Prototype-Based Semantic Labeling

## Concept
A query is compared against fixed label prototypes encoded as sentence vectors; the nearest prototype guides class selection.

## Where it is used
- nlp/classifier.py (CLAIM_PROTOTYPES + cosine similarity)

## Why it is used
- Works as a lightweight semantic classifier without needing full supervised fine-tuning.
- Easy to maintain and transparent for legal triage categories.

---

## 6. Intent Detection as Multi-Label Signal

## Concept
Intent extraction identifies user objective dimensions (not just case type), such as procedural help or document help.

## Where it is used
- nlp/classifier.py (INTENT_PATTERNS, intent_scores, intent_labels)

## Intents modeled
- procedural
- evidence
- forum
- timeline
- relief

## Why it is used
- Legal guidance quality depends on user intent, not only legal category.
- Improves downstream retrieval focus and response structure.

---

## 7. Urgency Detection (Risk-Aware Lexical Inference)

## Concept
Urgency classification detects temporal/risk criticality using lexical patterns and case context.

## Where it is used
- nlp/classifier.py (_detect_urgency)
- agents/langgraph_flow.py (safety escalation)

## Why it is used
- Legal triage requires time sensitivity handling.
- High-risk cases need immediate action prioritization before standard legal steps.

---

## 8. Priority Override Rules for Sensitive Misclassification Cases

## Concept
Post-classification overrides are rule-based corrections for known high-impact edge cases.

## Where it is used
- nlp/classifier.py (_apply_priority_overrides, _looks_like_child_labor, paid-resolution logic)

## Why it is used
- Reduces dangerous false positives/false routing in sensitive scenarios.
- Improves safety and fairness over purely statistical outputs.

---

## 9. Dense Retrieval (Vector Similarity Search)

## Concept
Dense retrieval maps queries and legal passages into embedding space and retrieves semantically close passages.

## Where it is used
- rag/retriever.py (LlamaIndex VectorStoreIndex + HuggingFaceEmbedding)

## Why it is used
- Legal queries are often paraphrased, narrative, and not keyword-exact.
- Dense retrieval captures semantic relevance beyond literal term overlap.

---

## 10. Retrieval Routing by Procedural Intent

## Concept
Intent-aware retrieval routing chooses between substantive judgments and procedural corpora.

## Where it is used
- rag/retriever.py (_is_procedural_intent, _merge_dual_results)

## Why it is used
- "What are my legal rights?" and "How do I file?" need different evidence sources.
- Route-aware retrieval increases practical usefulness of returned passages.

---

## 11. Query Rewriting / Expansion

## Concept
Query expansion creates multiple semantically related query variants to improve recall.

## Where it is used
- rag/retriever.py (_rewrite_queries)

## Why it is used
- User queries may omit legal framing terms.
- Rewrites recover relevant evidence for filing steps, forums, documents, and Maharashtra context.

---

## 12. Lexical Reranking with Phrase Matching

## Concept
After dense retrieval, lexical reranking boosts passages that contain important multi-word phrases.

## Where it is used
- rag/retriever.py (_extract_phrases, _boost_by_keywords_advanced)

## Why it is used
- Legal meaning often lives in phrase units, not isolated words.
- Phrase boosts improve precision for domain terms like wrongful termination, domestic violence, etc.

---

## 13. TF-IDF Weighting in Reranking

## Concept
Inverse Document Frequency (IDF) gives higher importance to rarer discriminative terms and lower importance to common terms.

## Where it is used
- rag/retriever.py (_calculate_idf, _boost_by_keywords_advanced)

## Formula

$$
\text{IDF}(w) = \log\left(\frac{N}{1 + df(w)}\right)
$$

## Why it is used
- Helps distinguish legally informative terms from generic text.
- Improves ranking quality for specific legal issue signals.

---

## 14. Domain Synonym Expansion

## Concept
Legal synonym mapping expands terms (for example wage -> salary/remuneration/dues).

## Where it is used
- rag/retriever.py (LEGAL_SYNONYMS, _expand_with_synonyms)

## Why it is used
- Users describe the same legal issue with varied vocabulary.
- Synonym expansion improves recall for colloquial and formal legal language variants.

---

## 15. Diversity-Aware Reranking

## Concept
Reranking penalizes near-duplicates and source over-concentration to produce a more diverse context bundle.

## Where it is used
- rag/retriever.py (_rerank_with_diversity)

## Why it is used
- Prevents redundant evidence in top-k results.
- Gives advisor stage a broader factual/contextual grounding set.

---

## 16. Grounded Generation (RAG-Conditioned Advice)

## Concept
Generated legal guidance is conditioned on retrieved passages + structured legal mapping, not only raw user text.

## Where it is used
- agents/langgraph_flow.py (StructuredAdvisor.generate, GrokClient.generate_structured)

## Why it is used
- Reduces hallucination risk versus free-form generation.
- Improves traceability by linking sections to retrieved evidence.

---

## 17. Schema-Constrained NLG (Structured JSON Output)

## Concept
Natural language generation is constrained to a strict schema (case scenario, steps, documents, courts, severity, flowchart, tts summary).

## Where it is used
- agents/langgraph_flow.py (Pydantic models + validation)

## Why it is used
- Produces consistent, machine-usable and UI-usable responses.
- Enforces output completeness and predictable section ordering.

---

## 18. Low-Grounding Detection and Controlled Fallback Generation

## Concept
When retrieval confidence is weak, generation mode is adjusted and deterministic fallback is used when needed.

## Where it is used
- agents/langgraph_flow.py (top score threshold, low_context_mode, deterministic_response)

## Why it is used
- Prevents overconfident legal guidance under weak evidence.
- Maintains reliability without requiring external LLM availability.

---

## 19. Safety-Aware NLP Layer (Risk Flagging + Escalation)

## Concept
Rule-driven risk detection augments generated advice with emergency-safe next actions and severity upgrades.

## Where it is used
- agents/langgraph_flow.py (_risk_flags, safety_node)

## Why it is used
- Critical legal-assistance requirement: immediate safety must supersede procedural advice.
- Adds explicit emergency pathways (for example 112, women helplines).

---

## 20. Citation Bucketing for Explainable NLP Outputs

## Concept
Retrieved passages are converted into section-wise citation bundles for explainability.

## Where it is used
- agents/langgraph_flow.py (_section_citations)

## Why it is used
- Makes the generated structure auditable.
- Helps users understand evidence behind each advisory section.

---

## 21. Follow-Up Question Generation for Missing Facts

## Concept
NLP follow-up generation identifies missing case facts and asks focused clarifying questions.

## Where it is used
- agents/langgraph_flow.py (_missing_facts_followups)

## Why it is used
- Legal reasoning quality depends on factual completeness.
- Promotes iterative enrichment instead of false certainty.

---

## 22. TTS Input Text Normalization (Post-NLP Surface Processing)

## Concept
Before speech synthesis, text is cleaned to remove markdown/control artifacts and normalize whitespace.

## Where it is used
- tts/speak.py (normalize_tts_text)

## Why it is used
- Ensures spoken output is natural and robust.
- Prevents reading of formatting artifacts that degrade user comprehension.

---

## 23. What is NOT used (important clarifier)

To avoid ambiguity, these are not core implemented NLP methods in the current codebase:
- No transformer fine-tuning on project-specific legal data.
- No heavy neural NER model training/fine-tuning pipeline.
- No dependency parsing or constituency parsing pipeline.
- No explicit stemming/lemmatization morphological pipeline.
- No cross-encoder reranker model; reranking is rule/score-boost based.

---

## 24. Summary: Why this NLP design works for this project

The project uses a practical hybrid NLP architecture:
1. ASR converts voice to text.
2. Hybrid classifier captures both semantics and deterministic legal cues.
3. Advanced RAG retrieval balances semantic recall with legal precision via reranking.
4. Structured grounded generation creates consistent legal guidance sections.
5. Safety NLP layer enforces emergency-aware behavior.
6. Explainability artifacts (citations, follow-ups, intents) support trust and auditability.

This combination is intentionally chosen for legal triage: robust to user language variation, controlled in high-risk contexts, and operational even with fallback-only execution.
