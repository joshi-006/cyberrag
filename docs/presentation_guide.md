# CyberRAG: Committee Presentation & Live Defense Guide

**15-Minute Defense Strategy & Live Demo Sequence**  
*Integrated Platform: Hybrid Graph RAG, Cross-Encoder NLI Verification, Provenance Preservation, and Context-Aware Contradiction Resolution*

---

## 1. Committee Presentation Structure (15 Minutes)

| Time | Slide / Topic | Key Message to Deliver |
| :--- | :--- | :--- |
| **00:00 – 02:30** | **The Problem with LLMs in CTI** | Generic LLMs and naive RAG pipelines suffer from hallucinations, ungrounded graph triples, false consensus from duplicated vendor reports, and temporal naivety when operational realities change. |
| **02:30 – 05:00** | **System Architecture** | Multi-stage pipeline: Cyber NLP (NER + clause-aware relation extraction) $\rightarrow$ Typed Knowledge Graph $\rightarrow$ Hybrid Dense/BM25/Graph Retrieval $\rightarrow$ DeBERTa-v3 Cross-Encoder NLI $\rightarrow$ Multi-publisher Claim Aggregator $\rightarrow$ Context Resolver $\rightarrow$ Structured Analyst Workbench. |
| **05:00 – 10:30** | **Live Interactive Platform Demo** | Run the FastAPI Analyst Workbench (`http://localhost:8080`) across core presets (Corroboration $\rightarrow$ Temporal Supersedence $\rightarrow$ Duplicate Suppression $\rightarrow$ Novel Intel Noise Rejection & Graph Expansion). |
| **10:30 – 12:30** | **Core Research Contributions** | 1) Clause-aware extraction with vendor & spatial trigger suppression; 2) Intra-publisher duplicate passage detection; 3) 4-tier CTI contradiction taxonomy (`TEMPORAL_SUPERSEDENCE`, `CAMPAIGN_DIVERGENCE`, `GENUINE_ATTRIBUTION_DISPUTE`, `AMBIGUOUS_CONFLICT`); 4) Structured provenance-grounded evidence reporting citing exact report and sentence IDs. |
| **12:30 – 15:00** | **Evaluation Metrics, Limitations & Q&A** | 1,400 graph nodes, 329 edges, 374 rule-extracted assertions (all unverified), 5,991 corpus segments, 85/85 passing unit tests across 16 test suites, and real-world AnnoCTR empirical benchmark evaluation. |

---

## 2. Live Demo Sequence (FastAPI Analyst Workbench)

### Launching the Workbench
Run the background server (or verify running at `http://127.0.0.1:8080`):
```bash
python demo/server.py
```
Open **`http://localhost:8080`** in any web browser.

The interface opens with a clean, warm off-white and royal cobalt blue aesthetic (`#f8fafc` background with white elevated cards), a responsive Vis.js knowledge graph canvas, real-time ReAct decision trace, DeBERTa NLI probability gauges, and a quick-select Evaluation Presets ribbon.

---

### Key Scenarios Walkthrough

#### Scenario 1: Corroborated Attribution (Preset 1)
* **Query / Preset**: `Preset 1: APT28 USES Zebrocy` (or `Does APT28 use Zebrocy?`)
* **Click**: `Investigate Claim`
* **What to Highlight to Committee**:
  1. **ReAct Decision Trace**: The agent expands aliases (`Fancy Bear`, `Pawn Storm`, `Sednit`), searches the typed knowledge graph, executes hybrid retrieval across dense/sparse pools, and passes candidate passages to the `EntityAwareEvidenceGate`.
  2. **Knowledge Graph Canvas**: The graph centers on `APT28` and `Zebrocy`, displaying the verified `USES` edge and linked neighboring techniques and malware.
  3. **Verification Verdict**: Resolves as **`SUPPORTED`** (`SINGLE_SOURCE` / `CORROBORATED` depending on multi-publisher corroboration).
  4. **Evidence Cards**: Show sentence-level provenance with report IDs, publication dates, and verbatim source sentences.

---

#### Scenario 2: Temporal Contradiction & Supersedence (Preset 2)
* **Query / Preset**: `Preset 2: Emotet USES TrickBot` (or `Does Emotet distribute TrickBot? Post-takedown 2021`)
* **Click**: `Investigate Claim`
* **What to Highlight to Committee**:
  1. **The Conflict**:
     - Earlier historical intelligence (October 2020, Intel 471 / Zscaler): `ENTAILMENT` (Emotet distributed TrickBot payloads).
     - Newly ingested post-seizure intelligence (May 20, 2021, Europol / CISA): `CONTRADICTION` (score > 0.99, Emotet infrastructure disrupted and ceased distributing TrickBot).
  2. **Context-Aware Contradiction Resolution**:
     - Standard NLI or RAG would label this a simple factual contradiction (`CONTRADICTED`) or average the probabilities.
     - CyberRAG inspects the publication timestamps, detects a **233-day delta** across the January 2021 infrastructure takedown, and outputs **`TEMPORAL_SUPERSEDENCE`** (`CONTRA_LATER`).
  3. **Synthetic Overlay Provenance**:
     - Emphasize to the committee that the post-takedown report carries the explicit provenance label `DEMO – SYNTHETIC (Europol/CISA)` to demonstrate temporal reasoning without corrupting the static historical baseline.

---

#### Scenario 3: Duplicate Evidence Suppression (Preset 3)
* **Query / Preset**: `Preset 3: TrickBot USES Emotet`
* **Click**: `Investigate Claim`
* **What to Highlight to Committee**:
  1. Multiple passages across syndicated vendor reports share near-identical text.
  2. Rather than counting repeated text as separate corroborating sources, the pipeline computes text hashes and passage normalizations, suppressing 3 duplicate passages.
  3. The final support count reflects genuine independent publisher consensus.

---

#### Scenario 4: Noise Rejection on Novel Threat Intel (Preset 4)
* **Query / Preset**: `Preset 4: Sandworm USES Cyclops Blink`
* **Click**: `Investigate Claim`
* **What to Highlight to Committee**:
  1. `Sandworm` and `Cyclops Blink` are absent from the historical 104-report baseline corpus.
  2. Standard cross-encoders without an entity gate would score irrelevant corpus passages (e.g. general botnet evasion) and hallucinate high-confidence false contradictions or false entailments.
  3. CyberRAG's `EntityAwareEvidenceGate` verifies entity co-occurrence, drops ungrounded candidates, and safely returns **`INSUFFICIENT_EVIDENCE`** with zero hallucinated claims.

---

#### Scenario 5: Live Custom Report Ingestion & Graph Expansion
* **Action**: Click `+ Ingest Threat Report` in the top header.
* **Feature**:
  1. Analysts can choose one of the 4 bundled preset CTI reports (`Europol Takedown`, `APT28 NATO Decoy`, `Emotet-Trickbot Partnership`, `Sandworm Energy Grid`) or paste an arbitrary unstructured CTI report.
  2. Click `Analyze & Extract Intelligence`.
  3. The Cyber NLP pipeline extracts entities and clause-aware relations, links MITRE ATT&CK IDs, and dynamically expands the in-memory graph without mutating baseline data on disk.

---

## 3. Anticipated Committee Questions & Defenses

### Q1: "Are the 374 assertions in your baseline graph human-verified?"
> **Defense**:  
> *"No, and we explicitly document this across the codebase and paper. All 374 historical assertions carry `trust_level: "unverified"` and `human_verified: false`. They were extracted via automated clause-aware syntactic rules from 62 AnnoCTR reports. Automated filtering must not be mistaken for human adjudication. All 374 graph assertions are automatically extracted and unreviewed; our formal double-annotator study measured relation extraction precision at about one third."*

### Q2: "What is your measured relation extraction precision on real text?"
> **Defense**:  
> *"The legacy file `outputs/reports/relation_sample_100_audit.json` was an automated rule-based self-check, not a measurement of accuracy. We measured real relation precision in a formal first-round double-annotator human study on a random sample of 100 relations (seed 42) drawn from the extractor's 192 candidate relations in `extracted_relations.json` across 36 reports—this is distinct from the 374 graph assertions. Two annotators evaluated each relation from the evidence sentence alone under original guidelines (which were later refined after the first round, though reported figures come strictly from original rules). Annotator A found 31/100 correct = 31.0% (95% Wilson CI 22.8%–40.6%), and Annotator B (revised sheet) found 40/100 correct = 40.0% (CI 30.9%–49.8%), changing 3 labels from an initial 37% submission with κ = 0.82. Agreement was 89/100 with substantial Cohen's κ = 0.76; 11 disagreements remain unadjudicated. Both agreed correct on 30 rows, bounding precision between 30% and 41%, with headline relation precision of about one third. On rows both labelled incorrect, the error mix was: relation not stated (~37), wrong entity (~19), reversed direction (2), and negated/hypothetical (1). The primary cause of false extractions is that entities appear in the same passage without the sentence stating the relation; enforcing a same-sentence constraint is future work. Breakdown by relation: TARGETS 8/27, USES 11/32, USES_TECHNIQUE 10/27, and EXPLOITS 1/3 (EXPLOITS too small to interpret). Extractor confidence scores do not separate correct from incorrect relations (mean 0.95 vs 0.91). System limits must be stated side by side: NER F1 0.4396 on AnnoCTR test (baseline 0.3497); relation precision about one third; 374 assertions automatically extracted and unreviewed; and no EXPLOITS edges in the baseline graph."*

### Q3: "Why use a dedicated DeBERTa cross-encoder instead of prompting a frontier LLM like GPT-4?"
> **Defense**:  
> *"Generative LLMs suffer from prompt sensitivity, position bias, and non-deterministic outputs. A dedicated cross-encoder (`cross-encoder/nli-deberta-v3-base`) performs joint cross-attention across premise and hypothesis simultaneously across all transformer layers. It outputs reproducible, sub-second semantic alignment scores completely offline with zero external API costs and zero data leakage."*

### Q4: "Can NLI softmax scores be interpreted as the probability of factual truth?"
> **Defense**:  
> *"No. Softmax scores measure linguistic entailment (whether premise text logically entails a hypothesis), not real-world historical ground truth. A vendor can state a false rumor with 100% grammatical confidence. CyberRAG therefore separates linguistic NLI alignment from contextual domain verification (publisher counts, dates, and contradiction resolution)."*

### Q5: "Why did you archive the v5 retrieval ablation study?"
> **Defense**:  
> *"During methodological review, we discovered a circular dependency in the exploratory v5 ablation: the hand-curated alias dictionary used to generate gold labels was also supplied to the expanded retrieval pipeline. In accordance with rigorous scientific practice, we archived that exploratory benchmark to `outputs/legacy/` and explicitly document non-circular relation evaluation (e.g., against AZERG) as future work."*

### Q6: "What does your Graph Retrieval actually do?"
> **Defense**:  
> *"Graph retrieval in CyberRAG is implemented as entity-linked assertion lookup and adjacent neighborhood traversal. It retrieves evidence-grounded relational triples connected to extracted query entities; it does not claim unconstrained multi-hop reasoning over arbitrary unindexed graph paths."*

### Q7: "Why do MCP and the Demo Workbench differ on 'Emotet USES TrickBot'?"
> **Defense**:  
> *"On the clean corpus the two interfaces can disagree. The workbench verifies Emotet uses TrickBot (7 supporting passages, 0 contradicting), because the agent loop's clause-aware gating filters out headline noise. Meanwhile, the MCP server flags a conflict (a genuine attribution dispute, 106-day delta) caused by an Intel 471 article title containing a negation ('Emotet takedown is not like the Trickbot takedown') which DeBERTa cross-encoder evaluates as a semantic contradiction. With the simulated takedown record added (`CTI_LOAD_DEMO_OVERLAY=1`), the workbench incorporates the May 20, 2021 Europol/CISA disruption statement and demonstrates temporal supersedence (233 days, `CONTRA_LATER`)."*

---

## 4. Evaluation Metrics Summary for Slides

* **Automated Test Suite**: **85 / 85 unit and integration tests passing** across 16 test suites (`python run_tests.py`), running 100% offline.
* **Corpus Scale**: 104 authentic threat reports, 5,991 indexed text segments, 2,698 paragraphs.
* **Knowledge Graph**: 1,400 nodes, 329 edges, 374 rule-extracted unverified assertions, 1,726 MITRE ATT&CK concepts.
* **AnnoCTR NER Benchmark (Official Test Split, Zero Overlap)**:
  - Hybrid NER F1 = **0.4396** (+28.5% relative gain over dictionary gazetteer at 0.3421).
  - High precision on Organizations (0.51) and Threat Actors (0.48); modest recall on complex Techniques (0.10).
* **Unseen CTI Test Paragraphs (Generalization Check)**:
  - Precision = **0.9388**, Recall = **0.9020**, F1 = **0.9200**, with **100.0% provenance metadata compliance**.
