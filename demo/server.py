"""
CyberRAG CTI Analyst Workbench - FastAPI Server
Exposes high-speed endpoints for the interactive CTI investigation workbench.
Coordinates the Autonomous ReAct Agent, Knowledge Graph, DeBERTa NLI, and local Qwen2.5 LLM.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import urllib.request
import json

# Ensure offline huggingface cache usage
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.graph_rag_pipeline import GraphRAGPipeline
from src.agent.retrieval_agent import CyberRAGAgent
from src.llm.llm_client import LocalLLMClient
from demo import api_v1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CyberRAG-Server")

app = FastAPI(title="ThreatIntel CTI Investigation Workbench", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global pipeline & agent instances
PIPELINE: Optional[GraphRAGPipeline] = None
AGENT: Optional[CyberRAGAgent] = None
LLM_CLIENT: Optional[LocalLLMClient] = None
KG_DB: Optional[Any] = None
LLM_EXTRACTOR: Optional[Any] = None


def get_pipeline_and_agent():
    global PIPELINE, AGENT, LLM_CLIENT, KG_DB, LLM_EXTRACTOR
    if PIPELINE is None or AGENT is None:
        logger.info("Initializing GraphRAG Pipeline, DeBERTa NLI Verifier, and CyberRAG Agent...")
        
        # Load real DeBERTa NLI Verifier at server startup
        try:
            from src.verification.evidence_verifier import CTIEvidenceVerifier
            verifier = CTIEvidenceVerifier()
            logger.info("CTIEvidenceVerifier (DeBERTa-v3) successfully loaded.")
        except Exception as e:
            logger.error(f"FATAL: Failed to load CTIEvidenceVerifier: {e}")
            raise RuntimeError(
                f"FATAL: DeBERTa NLI verifier failed to load at server startup: {e}. "
                f"Ensure cross-encoder/nli-deberta-v3-base is present in HuggingFace cache."
            )

        LLM_CLIENT = LocalLLMClient()
        PIPELINE = GraphRAGPipeline(verifier=verifier)
        AGENT = CyberRAGAgent(
            retriever=PIPELINE.retriever,
            verifier=verifier,
            context_resolver=PIPELINE.context_resolver,
            llm_client=LLM_CLIENT,
        )

        try:
            from src.graph.db import CTIKnowledgeGraphDB
            from src.graph.llm_extractor import LLMKnowledgeGraphExtractor
            KG_DB = CTIKnowledgeGraphDB()
            LLM_EXTRACTOR = LLMKnowledgeGraphExtractor(llm_client=LLM_CLIENT, db=KG_DB)
            logger.info("SQLite Knowledge Graph DB (data/cti_kg.db) and LLM Extractor connected.")
        except Exception as e:
            logger.warning(f"Could not initialize SQLite KG DB / LLM Extractor: {e}")

        # Pre-load post-takedown intelligence (Europol/CISA May 2021) to enable the temporal supersedence benchmark
        # Enabled by default in demo workbench, but strictly honours CTI_LOAD_DEMO_OVERLAY environment variable
        load_overlay = os.environ.get("CTI_LOAD_DEMO_OVERLAY", "1").strip().lower() in ("1", "true", "yes")
        if load_overlay:
            overlay_file = ROOT_DIR / "data" / "demo_overlay.jsonl"
            overlay_records = []
            if overlay_file.exists():
                for line in overlay_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        rec["synthetic"] = True
                        overlay_records.append(rec)
            if not overlay_records:
                overlay_records.append({
                    "report_id": "DEMO-CURATED-EUROPOL-2021-05-20-TAKEDOWN",
                    "segment_id": "DEMO-CURATED-EUROPOL-2021-05-20-TAKEDOWN::p1::s1",
                    "source": "Europol / CISA (Simulated Post-Takedown Demonstration Record)",
                    "publisher": "Europol / CISA",
                    "publication_date": "2021-05-20",
                    "text": (
                        "Following the global infrastructure disruption in January 2021, "
                        "Emotet no longer uses TrickBot payloads or distribution infrastructure across enterprise networks."
                    ),
                    "clean_text": (
                        "Following the global infrastructure disruption in January 2021, "
                        "Emotet no longer uses TrickBot payloads or distribution infrastructure across enterprise networks."
                    ),
                    "synthetic": True,
                })
            import numpy as np
            for rec in overlay_records:
                if not any(d.get("report_id") == rec["report_id"] for d in PIPELINE.retriever.dense.documents):
                    PIPELINE.retriever.dense.documents.append(rec)
                    text_val = rec.get("clean_text") or rec.get("text") or ""
                    PIPELINE.retriever.dense.texts.append(text_val)
                    new_emb = PIPELINE.retriever.dense.model.encode([text_val], normalize_embeddings=True)
                    PIPELINE.retriever.dense.embeddings = np.vstack([PIPELINE.retriever.dense.embeddings, new_emb])
            from src.retrieval import _tokens
            from rank_bm25 import BM25Okapi
            PIPELINE.retriever.hybrid.bm25 = BM25Okapi([_tokens(t) for t in PIPELINE.retriever.dense.texts])
            logger.info("Pre-loaded %d post-takedown demo overlay record(s) for temporal supersedence verification.", len(overlay_records))
        else:
            logger.info("Demo overlay disabled (CTI_LOAD_DEMO_OVERLAY=0); running workbench on clean production corpus.")

        logger.info("Pipeline and Agent ready!")
    return PIPELINE, AGENT


class QueryRequest(BaseModel):
    query: str
    scenario_id: Optional[str] = None


class ReportIngestionRequest(BaseModel):
    title: str = "Uploaded CTI Report"
    source: str = "Analyst Submission"
    publication_date: str = "2024-01-01"
    text: str


class BriefExportRequest(BaseModel):
    query: str
    status: str = "SUPPORTED"
    confidence_score: float = 0.8
    consensus_score: Optional[float] = None
    source_diversity: int = 1
    analyst_decision: str = "ACCEPT"  # ACCEPT, QUALIFY, NEEDS_MORE_EVIDENCE
    analyst_notes: str = ""
    source_reliability: str = "B - Usually reliable"
    info_credibility: str = "2 - Probably true"
    tlp_label: str = "TLP:GREEN"  # TLP:CLEAR, TLP:GREEN, TLP:AMBER, TLP:RED
    supporting_evidence: List[Dict[str, Any]] = []
    contradicting_evidence: List[Dict[str, Any]] = []
    conflict_resolution: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# /api/v1: VerificationService (shared with the MCP server) + analyst tools.
# Built in a background thread after the agent pipeline so the workbench is
# usable immediately; it reuses the already-loaded DeBERTa verifier.
# ---------------------------------------------------------------------------
import threading

_SERVICE: Dict[str, Any] = {"svc": None, "error": None}
_SERVICE_LOCK = threading.Lock()


def _build_verification_service() -> None:
    try:
        from src.core.bootstrap import build_service
        from src.core.config import load_settings

        settings = load_settings(load_demo_overlay=True)
        shared = getattr(PIPELINE, "verifier", None)
        if shared is not None and getattr(shared, "model_name", None) != settings.nli_model:
            shared = None
        svc = build_service(settings, verifier=shared)
        with _SERVICE_LOCK:
            _SERVICE["svc"] = svc
        logger.info("VerificationService ready for /api/v1 (demo overlay records: %s).", svc.overlay_loaded)
    except Exception as exc:  # keep the workbench up even if v1 cannot start
        logger.exception("VerificationService failed to start; /api/v1 will return 503")
        with _SERVICE_LOCK:
            _SERVICE["error"] = str(exc)


def _service_provider():
    with _SERVICE_LOCK:
        svc, err = _SERVICE["svc"], _SERVICE["error"]
    if svc is not None:
        return svc
    raise api_v1.ServiceWarmingUp(f"Verification service failed to start: {err}" if err else
                                  "Verification service is still loading models; retry in a few seconds.")


api_v1.set_service_provider(_service_provider)
app.include_router(api_v1.router)


@app.on_event("startup")
async def startup_event():
    # Warm up pipeline in background/on startup
    get_pipeline_and_agent()
    threading.Thread(target=_build_verification_service, name="v1-service", daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(__file__).parent / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(
        content=index_path.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/health")
async def check_health():
    pipeline, agent = get_pipeline_and_agent()
    llm_connected = False
    active_model = "Unavailable"
    active_port = None

    for port in (1235, 1234):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"User-Agent": "CyberRAG/2.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", [])]
                if models:
                    llm_connected = True
                    active_model = models[0]
                    active_port = port
                    break
        except Exception:
            pass

    verifier_ready = (pipeline is not None and getattr(pipeline, "verifier", None) is not None)
    db_stats = KG_DB.get_stats() if KG_DB is not None else {}
    return {
        "status": "online",
        "verifier_status": "ready" if verifier_ready else "failed",
        "verifier_model": "cross-encoder/nli-deberta-v3-base",
        "llm_connected": llm_connected,
        "active_model": active_model,
        "active_port": active_port,
        "database": "SQLite (data/cti_kg.db)",
        "indexed_segments": len(pipeline.retriever.dense.documents),
        "indexed_entities": db_stats.get("total_entities") or len(pipeline.retriever.graph.entity_lookup),
        "indexed_assertions": db_stats.get("total_relationships") or len(pipeline.retriever.graph.assertions),
        "llm_extractor": "ready" if LLM_EXTRACTOR is not None else "unavailable",
    }


QUERY_VERB_STOPWORDS = {
    "use", "uses", "used", "using",
    "target", "targets", "targeted", "targeting",
    "exploit", "exploits", "exploited", "exploiting",
    "does", "did", "is", "was", "has", "have", "had", "are", "were"
}


@app.post("/api/investigate")
async def run_investigation(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    pipeline, agent = get_pipeline_and_agent()

    logger.info(f"Executing Agent Investigation for query: '{query}'")
    report = agent.run(query)

    # Extract local subgraph around query entities for UI visualization
    graph_retriever = pipeline.retriever.graph
    raw_matched = graph_retriever.match_entities(query)
    # Filter out query verbs so words like 'use' are not matched as stray entity nodes
    matched_entities = [(name, eid) for name, eid in raw_matched if name.lower() not in QUERY_VERB_STOPWORDS]
    
    subgraph_nodes: Dict[str, Dict[str, Any]] = {}
    subgraph_edges: List[Dict[str, Any]] = []

    # 1. Query SQLite Knowledge Graph Database first
    if KG_DB is not None:
        for name, eid in matched_entities:
            neighborhood = KG_DB.get_neighborhood(eid, max_hops=1)
            for node in neighborhood.get("nodes", []):
                nid = node["id"]
                if nid not in subgraph_nodes:
                    subgraph_nodes[nid] = {
                        "id": nid,
                        "label": node.get("label", nid),
                        "group": "matched_entity" if nid == eid else "neighbor",
                        "type": node.get("category", "Entity"),
                    }
            for edge in neighborhood.get("edges", []):
                edge_id = f"{edge['from']}->{edge['label']}->{edge['to']}"
                if not any(e["id"] == edge_id for e in subgraph_edges):
                    subgraph_edges.append({
                        "id": edge_id,
                        "from": edge["from"],
                        "to": edge["to"],
                        "label": edge["label"],
                        "confidence": edge.get("confidence", 0.85),
                    })

    # 2. Fallback to in-memory graph index if DB query produced no edges
    if not subgraph_edges:
        for name, eid in matched_entities:
            subgraph_nodes[eid] = {
                "id": eid,
                "label": name.upper(),
                "group": "matched_entity",
                "type": eid.split("::")[0] if "::" in eid else "Entity",
            }
            # Add adjacent assertions
            for assertion in graph_retriever.entity_index.get(eid, [])[:8]:
                sub_id = assertion.get("subject_id", "")
                obj_id = assertion.get("object_id", "")
                rel = assertion.get("predicate") or assertion.get("relation") or "RELATES_TO"
                
                if sub_id and sub_id not in subgraph_nodes:
                    sub_label = sub_id.split("::")[-1] if "::" in sub_id else sub_id
                    subgraph_nodes[sub_id] = {
                        "id": sub_id,
                        "label": sub_label,
                        "group": "neighbor",
                        "type": sub_id.split("::")[0] if "::" in sub_id else "Entity",
                    }
                if obj_id and obj_id not in subgraph_nodes:
                    obj_label = obj_id.split("::")[-1] if "::" in obj_id else obj_id
                    subgraph_nodes[obj_id] = {
                        "id": obj_id,
                        "label": obj_label,
                        "group": "neighbor",
                        "type": obj_id.split("::")[0] if "::" in obj_id else "Entity",
                    }
                
                edge_id = f"{sub_id}->{rel}->{obj_id}"
                if not any(e["id"] == edge_id for e in subgraph_edges):
                    subgraph_edges.append({
                        "id": edge_id,
                        "from": sub_id,
                        "to": obj_id,
                        "label": rel,
                        "confidence": assertion.get("confidence", 0.8),
                    })

    # Serialize execution trace
    serializable_trace = []
    for step in report.execution_trace:
        serializable_trace.append({
            "step_index": step.step_index,
            "state": step.state.value if hasattr(step.state, "value") else str(step.state),
            "action": getattr(step, "action", "") or step.thought,
            "decision_outcome": getattr(step, "decision_outcome", "") or step.tool_output_summary,
            "retry_or_termination_reason": getattr(step, "retry_or_termination_reason", ""),
            "thought": getattr(step, "action", "") or step.thought,
            "tool_name": step.tool_name,
            "tool_input": step.tool_input,
            "tool_output_summary": step.tool_output_summary,
        })

    # Clean supporting evidence for JSON serialization
    supporting_cleaned = []
    for item in report.supporting_evidence:
        supporting_cleaned.append({
            "report_id": item.get("report_id"),
            "segment_id": item.get("segment_id"),
            "source": item.get("source"),
            "publication_date": item.get("publication_date"),
            "text": item.get("text") or item.get("evidence"),
            "p_entail": round(item.get("p_entail", 0.0), 4),
            "p_contra": round(item.get("p_contra", 0.0), 4),
            "p_neutral": round(item.get("p_neutral", 0.0), 4),
            "nli_label": item.get("nli_label", "ENTAILMENT"),
        })

    contradicting_cleaned = []
    for item in report.contradicting_evidence:
        contradicting_cleaned.append({
            "report_id": item.get("report_id"),
            "segment_id": item.get("segment_id"),
            "source": item.get("source"),
            "publication_date": item.get("publication_date"),
            "text": item.get("text") or item.get("evidence"),
            "p_entail": round(item.get("p_entail", 0.0), 4),
            "p_contra": round(item.get("p_contra", 0.0), 4),
            "p_neutral": round(item.get("p_neutral", 0.0), 4),
            "nli_label": item.get("nli_label", "CONTRADICTION"),
        })

    status = report.status
    hedged_only = (status == "WEAK_SUPPORT")
    if status in ("VERIFIED", "WEAK_SUPPORT"):
        status = "SUPPORTED"

    return {
        "query": report.query,
        "status": status,
        "hedged_only": hedged_only,
        "raw_status": report.status,
        "consensus_evidence_score": round(report.confidence_score, 4),
        "confidence_score": round(report.confidence_score, 4),  # Deprecated alias for backwards compatibility
        "source_diversity": report.source_diversity,
        "matched_entities": [{"id": eid, "name": name} for name, eid in matched_entities],
        "state_history": [s.value if hasattr(s, "value") else str(s) for s in report.state_history],
        "execution_trace": serializable_trace,
        "supporting_evidence": supporting_cleaned,
        "contradicting_evidence": contradicting_cleaned,
        "conflict_resolution": report.conflict_resolution,
        "narrative": report.narrative,
        "subgraph": {
            "nodes": list(subgraph_nodes.values()),
            "edges": subgraph_edges,
        },
    }


class CheckClaimsRequest(BaseModel):
    queries: List[str]


@app.post("/api/check_claims")
async def check_claims(req: CheckClaimsRequest):
    """Re-verify watched claims with the same agent pipeline as /api/investigate.

    Uses the live workbench index, so reports added with /api/ingest_report are
    taken into account. Returns a compact verdict per claim (max 20).
    """
    queries = [q.strip() for q in req.queries if isinstance(q, str) and q.strip()][:20]
    if not queries:
        raise HTTPException(status_code=400, detail="queries must contain 1-20 non-empty claims")
    _, agent = get_pipeline_and_agent()
    out = []
    for q in queries:
        try:
            r = agent.run(q[:500])
            out.append({
                "query": q, "status": r.status, "supporting": len(r.supporting_evidence),
                "contradicting": len(r.contradicting_evidence), "source_diversity": r.source_diversity,
                "score": round(float(r.confidence_score or 0), 4),
            })
        except Exception as exc:  # one bad claim must not fail the whole re-check
            logger.warning("check_claims failed for %r: %s", q, exc)
            out.append({"query": q, "status": "ERROR", "error": str(exc)[:200]})
    return {"results": out}


@app.post("/api/ingest_report")
async def ingest_report(req: ReportIngestionRequest):
    """Dynamically ingest a threat report into in-memory Knowledge Graph and Retriever."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Report text cannot be empty")

    pipeline, agent = get_pipeline_and_agent()
    from src.ingestion import LiveReportProcessor
    processor = LiveReportProcessor(pipeline=pipeline)

    rep_id = f"UPLOAD-{abs(hash(text)) % 100000:05d}"
    extraction = processor.process_report(
        raw_text=text,
        report_id=rep_id,
        source=req.source,
        pub_date=req.publication_date,
        title=req.title,
    )

    merged = processor.merge_with_baseline_graph(extraction)

    # Persist extracted knowledge into SQLite DB via LLM Extractor
    if LLM_EXTRACTOR is not None:
        try:
            llm_res = LLM_EXTRACTOR.extract_from_text(
                text=text,
                metadata={"report_id": rep_id, "source": req.source, "publication_date": req.publication_date},
                commit_to_db=True,
            )
            logger.info("LLM Extractor committed %d entities and %d relations to SQLite KG",
                        llm_res.get("total_entities", 0), llm_res.get("total_relationships", 0))
        except Exception as e:
            logger.warning("LLM extraction warning during report ingestion: %s", e)

    # Ingest sentences into dense & hybrid retriever
    sentences = extraction.get("sentences", [])
    if sentences:
        new_texts = []
        for sent in sentences:
            seg = {
                "report_id": rep_id,
                "segment_id": sent.sentence_id,
                "source": req.source,
                "publication_date": req.publication_date,
                "text": sent.text,
                "clean_text": sent.text,
            }
            pipeline.retriever.dense.documents.append(seg)
            pipeline.retriever.dense.texts.append(sent.text)
            new_texts.append(sent.text)

        import numpy as np
        new_embs = pipeline.retriever.dense.model.encode(new_texts, normalize_embeddings=True)
        pipeline.retriever.dense.embeddings = np.vstack([pipeline.retriever.dense.embeddings, new_embs])

        from src.retrieval import _tokens
        from rank_bm25 import BM25Okapi
        pipeline.retriever.hybrid.bm25 = BM25Okapi([_tokens(t) for t in pipeline.retriever.dense.texts])

    return {
        "status": "success",
        "report_id": rep_id,
        "title": req.title,
        "total_sentences": len(sentences),
        "extracted_entities": len(extraction.get("unique_entities", [])),
        "extracted_relations": len(extraction.get("relations", [])),
        "new_nodes_added": merged.get("new_nodes_count", 0),
        "new_edges_added": merged.get("new_edges_count", 0),
    }


@app.post("/api/export_brief")
async def export_evidence_brief(req: BriefExportRequest):
    """
    Export evidence brief with:
    - Target claim & cutoff date
    - Automated verdict & consensus score
    - Human analyst decision (accept/qualify/needs_more_evidence)
    - Analyst notes & caveats
    - Admiralty source reliability & credibility ratings
    - TLP sharing label
    - Sentence-level provenance citations
    """
    from datetime import datetime
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    dates = [
        e.get("publication_date") for e in req.supporting_evidence + req.contradicting_evidence
        if e.get("publication_date")
    ]
    cutoff_date = max(dates) if dates else "Undated"

    clean_q = "".join(c if c.isalnum() else "_" for c in req.query)[:35].strip("_")
    filename = f"CTI_Brief_{clean_q}_{req.tlp_label.replace(':', '_')}.md"

    md_lines = [
        f"# CTI EVIDENCE BRIEF: {req.query}",
        f"**Classification**: `{req.tlp_label}`",
        f"**Admiralty Code**: `{req.source_reliability[:1]}{req.info_credibility[:1]}` ({req.source_reliability} / {req.info_credibility})",
        f"**Generated**: {now_str}",
        f"**Corpus Cutoff Date**: {cutoff_date}",
        "",
        "---",
        "",
        "## 1. Executive Summary & Verdict",
        f"- **Automated Intelligence Verdict**: **{req.status.upper()}**",
        f"- **Rule-Based Evidence Score**: {round(req.confidence_score * 100)}% (Rule-based, uncalibrated heuristic over evidence found; not a statistical probability)",
        f"- **Independent Publishers**: {req.source_diversity}",
        f"- **Evidence Volume**: {len(req.supporting_evidence)} supporting passage(s), {len(req.contradicting_evidence)} contradicting passage(s)",
        "",
        "## 2. Human Analyst Review & Adjudication",
        f"- **Analyst Review Decision**: **{req.analyst_decision.upper()}** (Human analyst review step; does not validate underlying corpus assertions)",
        f"- **Source Reliability**: {req.source_reliability}",
        f"- **Information Credibility**: {req.info_credibility}",
        "- **Analyst Assessment Notes**:",
        f"  > {req.analyst_notes if req.analyst_notes.strip() else '_No additional analyst notes recorded._'}",
        "",
    ]

    if req.conflict_resolution and req.conflict_resolution.get("resolution_status"):
        cr = req.conflict_resolution
        md_lines.extend([
            "## 3. Conflict Resolution & Temporal Dynamics",
            f"- **Resolution Status**: `{cr.get('resolution_status')}`",
            f"- **Temporal Delta**: {cr.get('delta_days', 'N/A')} days",
            f"- **Analysis**: {cr.get('explanation', 'N/A')}",
            "",
        ])

    md_lines.append("## 4. Provenance-Grounded Evidence Citations\n")
    if req.supporting_evidence:
        md_lines.append(f"### Supporting Evidence ({len(req.supporting_evidence)})\n")
        for i, e in enumerate(req.supporting_evidence, 1):
            src = e.get("source") or "Unknown Publisher"
            pdate = e.get("publication_date") or "Undated"
            seg = e.get("segment_id") or e.get("report_id") or "N/A"
            txt = (e.get("text") or "").replace("\n", " ").strip()
            md_lines.append(f"{i}. **{src}** ({pdate}) | Segment ID: `{seg}`")
            md_lines.append(f"   > \"{txt}\"")
            if "p_entail" in e:
                md_lines.append(f"   *(DeBERTa-v3 NLI Entailment: {e['p_entail']:.3f})*")
            md_lines.append("")

    if req.contradicting_evidence:
        md_lines.append(f"### Contradicting Evidence ({len(req.contradicting_evidence)})\n")
        for i, e in enumerate(req.contradicting_evidence, 1):
            src = e.get("source") or "Unknown Publisher"
            pdate = e.get("publication_date") or "Undated"
            seg = e.get("segment_id") or e.get("report_id") or "N/A"
            txt = (e.get("text") or "").replace("\n", " ").strip()
            md_lines.append(f"{i}. **{src}** ({pdate}) | Segment ID: `{seg}`")
            md_lines.append(f"   > \"{txt}\"")
            if "p_contra" in e:
                md_lines.append(f"   *(DeBERTa-v3 NLI Contradiction: {e['p_contra']:.3f})*")
            md_lines.append("")

    md_lines.extend([
        "---",
        "*Exported from ThreatIntel CTI Investigation Workbench · Complete Sentence-Level Provenance Audit Trail Preserved*",
        "*Caveat: The evidence score is a rule-based heuristic over retrieved sentences, not a calibrated probability. Underlying knowledge graph relations are automatically extracted and unreviewed (human double-annotator precision ~1/3). This brief records an analyst review step over raw evidence citations.*",
    ])

    return {
        "status": "success",
        "filename": filename,
        "markdown": "\n".join(md_lines),
        "tlp": req.tlp_label,
    }


def _open_browser_when_ready(port: int) -> None:
    """Polls /api/health until the server and models are fully initialized before opening browser."""
    import time
    import webbrowser

    # Disable auto-open if either OPEN_BROWSER=0 or NO_BROWSER=1 is set
    if os.environ.get("OPEN_BROWSER", "1").strip().lower() in ("0", "false", "no"):
        return
    if os.environ.get("NO_BROWSER", "0").strip().lower() in ("1", "true", "yes"):
        return

    health_url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + 180.0
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "CyberRAG-Launcher"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    time.sleep(0.3)
                    webbrowser.open(f"http://127.0.0.1:{port}")
                    return
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    import uvicorn
    import threading

    port = int(os.environ.get("PORT", 8080))
    print(f"\n==================================================================")
    print(f"  CYBERRAG CTI ANALYST WORKBENCH LAUNCHING")
    print(f"  URL: http://127.0.0.1:{port}")
    print(f"==================================================================\n")
    threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()
    uvicorn.run("demo.server:app", host="127.0.0.1", port=port, reload=False)
