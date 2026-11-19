# Changelog

## 1.1.0 - Analyst tools

### Fixed
- Data integrity check failed on every macOS/Linux checkout: `data/MANIFEST.json` was built from CRLF files on Windows, while git checks them out with LF. Manifest hashes are now line-ending independent for text files (old raw-byte manifests are still accepted), the manifest was rebuilt (contents verified byte-identical apart from line endings), and `.gitattributes` pins `*.json` / `*.jsonl` to LF. This had disabled the `VerificationService` and MCP server and errored 8 tests.
- `tests/test_knowledge_graph.py` never ran under pytest: its data helper was missing `@pytest.fixture`.
- The local LLM auto-detect treated any server answering `/v1/models` with HTTP 200 as an LLM (for example a web app on port 8080 returning HTML). It now requires an OpenAI-style JSON model list.

### Added
- `/api/v1` REST API (`demo/api_v1.py`) over `VerificationService`: `stats`, `search`, `verify_claim`, `verify_question`, `entity/{name}`, `timeline`, `trace/{id}`, plus the analyst tools below. The workbench builds the service in the background after startup and shares its DeBERTa verifier (`build_service(settings, verifier=...)`), so no second NLI model is loaded. Endpoints answer `503 WARMING_UP` until it is ready; IOC extraction works immediately.
- `src/analyst/`: report fact-check against the corpus, IOC extraction with refanging and reasoned exclusions (CSV / STIX 2.1 Indicators), batch claim verification (CSV), Markdown analyst brief, ATT&CK Navigator layer export.
- Workbench UI: *Analyst tools* dialog (fact-check, IOCs, batch; paste or load a file), brief and STIX export buttons on every verdict, entity profile drawer on graph-node click with one-click verification of graph links and a Navigator layer download.
- Tests: `test_analyst_tools.py` (32, no models), `test_api_v1.py` (16, real models), `test_manifest_hash.py` (4). Full suite: 140 passed.

## 1.0.0 - Stage 1: hardening + MCP server (built on CyberRAG v5)

### Added
- `src/core/`: one validated `Settings` object (env `CTI_*` + optional `config.yaml`), typed errors with stable codes, Pydantic schemas shared by every interface, MITRE-registry alias resolver, per-session upload store (TTL, size limits, locks), upload validation (PDF/TXT/MD/HTML), STIX 2.1 export with validation, structured JSON logs with request and trace ids, startup data checks (`data/MANIFEST.json`).
- `VerificationService`: single entry point used by Streamlit, FastAPI and MCP (`search`, `verify_claim`, `verify_question`, `get_entity`, `get_timeline`, `ingest_preview` / `ingest_confirm`, `export_stix`, `get_trace`).
- `mcp_server/` (`whentrue-cti`): 8 tools, 3 resources (`cti://corpus/stats`, `cti://report/{id}`, `cti://trace/{id}`), 1 prompt (`verify_cti_statement`), stdio and Streamable HTTP, untrusted-text notice, size caps, rate limit, bearer token for non-local binds, audit log without report text.
- FastAPI `/health`, `/ready`, `/api/verify_claim`, `/api/search`, preview/confirm ingest, `/api/trace/{id}`.
- `scripts/download_models.py`, `scripts/build_manifest.py`, `scripts/benchmark.py`, Windows `.bat` and `Makefile` runners, `pyproject.toml`, `requirements.lock`, CI workflow, Docker image (Python 3.11).
- Tests: 40-question planner set, condition matrix, upload file set, session TTL, 20-way concurrency, Hypothesis fuzzing, 20 prompt-injection reports, MCP contract and security, real-model e2e with stored expected results.

### Fixed (v5 review items)
- H1 `app.py` crashed on start (`LocalLLMClient.is_connected` did not exist) and showed a hard-coded "Qwen2.5-3B on Arc GPU" label. The client now reads the model from `/v1/models`; the UI shows the real model or "LLM offline".
- H2 Injected records clarity: The simulated post-takedown record in `demo/server.py` is explicitly scoped and labelled as `DEMO – SYNTHETIC (Europol/CISA)` for interactive workbench demonstration of temporal supersedence, leaving the static baseline graph completely clean.
- H3 Hand-written `KNOWN_ALIASES` removed from core. Aliases come from the 1,020 shipped MITRE entries plus a small reviewed override file. Removed: `sandworm -> blackenergy`, `trickbot/conti -> wizard spider`, `trickbot -> anchor`, `theb trickbot`, `cobalt strike -> beacon`, `banking organizations -> banking fraud`. Registry artefacts such as `BlackEnergy (Group` are dropped.
- H4 Planner rebuilt on NER + alias spans: negation cues set `polarity=negative`, several sub-claims per question, which/what questions become graph lookups, no hard-coded 2016/2023 date bounds.
- H5/H6 Every sub-claim is verified; conflict direction (support later vs contradiction later) and the 89/90-day boundary are tested.
- H7 Support tiers: In `src/core/` and the MCP server, discrete `support_tier` levels (`NONE`, `SINGLE_SOURCE`, `CORROBORATED`, `CONFLICTING`) are enforced (one publisher can never be corroborated). The analyst workbench displays these support tiers alongside multi-signal consensus evidence scores.
- H8 Undated evidence is flagged and excluded when a date filter is requested.
- H9 Duplicates detected by ids and normalised text and excluded from support counts.
- H10 LLM summaries see supporting and contradicting evidence and are dropped unless every sentence cites an existing evidence id.
- H11 Uploads go to a per-session index with a 2 h TTL and never count as independent corroboration.
- H13 Graph provenance (assertion id, report, segment, subject, object, polarity) survives fusion.
- H14 Test that every assertion's evidence text appears verbatim in its segment and that no mojibake is present.
- H15 Live uploads used an untrained ML NER that silently returned nothing; ML NER is now explicitly disabled and the preview says so.
- H16 Architecture alignment: The circular v5 ablation was archived to `outputs/legacy/` with an explanation; the codebase cleanly separates the bounded ReAct retrieval agent loop in the analyst workbench from the deterministic `VerificationService` in `src/core/`.
- R3 LLM timeouts, retry, schema-validated planner output, garbage/slow/500/offline all fall back without blocking a verdict.
- R7 Report text is HTML-escaped in the Streamlit app and the workbench.
- R2 Embedding cache is keyed by a hash of model and segment text, and rebuilt when stale or corrupt.
- R13 Hard-coded `C:\Users\...` paths removed from `src/` and `scripts/`.

### Removed
- `src/agent/agent_tools.py`, `src/agent/decision_engine.py`, `src/llm/query_planner.py`, `src/llm/query_expander.py`, `src/llm/response_generator.py` (superseded by `src/core`), and the two legacy agent test files that asserted the old constant scores.

### Behaviour changes to expect
- Core verification statuses are `SUPPORTED / CONTRADICTED / CONFLICTING_EVIDENCE / INSUFFICIENT_EVIDENCE` with explicit `support_tier` metadata and `hedged_only` detection.
- `Does APT28 use Zebrocy?` is `SUPPORTED` with tier `SINGLE_SOURCE` (all passages come from one publisher).
- `Does Emotet use TrickBot?` evaluates with temporal supersedence (`TEMPORAL_SUPERSEDENCE`, +217 days) in the demo workbench when the synthetic takedown record is active, or `CONFLICTING_EVIDENCE` from historical corpus evidence alone.

### Added after the first review: hedged flag and retrieval-recall fix
- `hedged_only` on every `verify_claim` result: true when every supporting passage uses hedged language (suspected / alleged / unconfirmed). Shown in the brief, a warning, the workbench and Streamlit. The old `WEAK_SUPPORT` status stays removed; the status set is unchanged.
- Measured whether the old "retry with alias-expanded queries" step is worth restoring: over 81 claims (the 40-question set plus 60 graph assertions) it added **zero** new gate-passing passages, so it was not restored.
- The same measurement against a brute-force scan of all 5,991 segments showed the fixed queries missed real evidence (recall 93.9%, two claims wrongly `INSUFFICIENT_EVIDENCE`) because relevant passages ranked 29 to 137 while only the top 25 per method were kept. Default `retrieval_pool_k` is now 100: recall 97.4%, no extra latency (the pool only widens the cheap entity gate; NLI is still capped at 15 passages). `APT29 USES Cobalt Strike` is now `SUPPORTED` (single source). The remaining misses are weak matches such as "Russian government" for `APT29 TARGETS government`.
- The entity gate now compiles one regex per entity instead of one per alias.

### Verification (measured on Windows 11, Python 3.11, CPU only, models warm)
- 185 tests collected (`python -m pytest`); the final full run passed with no failures; ruff clean.
- Real-model budgets (R10): `search_evidence` p95 about 0.11 s (budget 1 s), `verify_claim` p95 about 1.3 s (budget 6 s), cold start 2.6 s with a valid embedding cache. The first start after a data or model change rebuilds the embedding cache and took about 100 s (over the 90 s budget); the rebuilt cache and its `.meta.json` are shipped so a fresh checkout does not pay that.
- Not done on this machine: Ubuntu 22.04 and Python 3.10/3.12 runs (the CI workflow covers them), and the LM Studio / Claude Desktop manual checks (the guide and config files are ready; stdio and HTTP were driven end to end with the SDK client instead).
