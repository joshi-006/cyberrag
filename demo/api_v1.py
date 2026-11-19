"""Versioned REST API for the workbench: ``/api/v1/...``.

Thin HTTP layer over ``VerificationService`` (the same core the MCP server and
the tests use) plus the analyst tools in ``src.analyst``. The service is
attached with :func:`set_service_provider`, so the router can be mounted in the
workbench or in a test app without loading the legacy agent pipeline.

Endpoints are plain ``def`` so FastAPI runs the CPU-bound model work in its
thread pool instead of blocking the event loop.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from src.core.errors import CTIError

router = APIRouter(prefix="/api/v1", tags=["v1"])

_provider: Optional[Callable[[], Any]] = None
_STATUS = {
    "INVALID_INPUT": 400, "UNKNOWN_RELATION": 400, "UPLOAD_REJECTED": 400, "NO_EXTRACTABLE_TEXT": 422,
    "ENTITY_NOT_FOUND": 404, "NO_EVIDENCE": 404, "TOKEN_EXPIRED": 410, "RATE_LIMITED": 429,
    "VERIFIER_UNAVAILABLE": 503, "LLM_UNAVAILABLE": 503,
}


class ServiceWarmingUp(Exception):
    pass


def set_service_provider(fn: Callable[[], Any]) -> None:
    """``fn`` returns the VerificationService, or raises ServiceWarmingUp while it loads."""
    global _provider
    _provider = fn


def _svc():
    if _provider is None:
        raise ServiceWarmingUp("verification service not configured")
    return _provider()


def _call(fn: Callable[[], Any]):
    try:
        out = fn()
    except ServiceWarmingUp as exc:
        return JSONResponse(status_code=503, content={"code": "WARMING_UP", "message": str(exc) or "Service is loading models; retry in a few seconds.", "hint": "Retry shortly."})
    except CTIError as exc:
        return JSONResponse(status_code=_STATUS.get(exc.code, 500), content=exc.to_dict())
    if hasattr(out, "model_dump"):
        return out.model_dump()
    return out


def _safe_filename(name: str, ext: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "export").strip("._")[:80] or "export"
    return f"{base}.{ext}"


def _download(body: str, filename: str, media_type: str) -> Response:
    return Response(content=body, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ------------------------------------------------------------------ models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    publishers: Optional[List[str]] = None


class ClaimRequest(BaseModel):
    subject: str
    relation: str
    object: str
    polarity: str = "positive"
    as_of_date: Optional[str] = None
    date_from: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str


class StixRequest(BaseModel):
    verification_ids: Optional[List[str]] = None
    question: Optional[str] = None


class TextRequest(BaseModel):
    text: str = Field(..., max_length=2_000_000)
    title: str = "Analyst report"
    publisher: str = "Analyst Submission"
    pub_date: str = ""
    include_reference_domains: bool = False


class BatchRequest(BaseModel):
    text: str = Field(..., max_length=200_000)


# ------------------------------------------------------------------ core
@router.get("/stats")
def stats():
    return _call(lambda: _svc().stats())


@router.post("/search")
def search(req: SearchRequest):
    return _call(lambda: _svc().search(req.query, top_k=req.top_k, date_from=req.date_from,
                                       date_to=req.date_to, publishers=req.publishers))


@router.post("/verify_claim")
def verify_claim(req: ClaimRequest):
    return _call(lambda: _svc().verify_claim(req.subject, req.relation, req.object, as_of_date=req.as_of_date,
                                             polarity=req.polarity, date_from=req.date_from))


@router.post("/verify_question")
def verify_question(req: QuestionRequest):
    return _call(lambda: _svc().verify_question(req.question))


@router.get("/entity/{name}")
def entity(name: str):
    return _call(lambda: _svc().get_entity(name))


@router.get("/timeline")
def timeline(subject: str, object: str, relation: Optional[str] = None):  # noqa: A002
    return _call(lambda: _svc().get_timeline(subject, object, relation=relation))


@router.get("/trace/{verification_id}")
def trace(verification_id: str):
    return _call(lambda: _svc().get_trace(verification_id))


# ------------------------------------------------------------------ exports
@router.post("/export/stix")
def export_stix(req: StixRequest):
    """STIX 2.1 bundle for verification ids, or for every sub-claim of a question."""
    def run():
        svc = _svc()
        ids = list(req.verification_ids or [])
        if not ids and req.question:
            q = svc.verify_question(req.question)
            ids = [r["verification_id"] for r in q.get("results", [])]
            if not ids:
                from src.core.errors import InvalidInputError
                raise InvalidInputError("The question did not contain a checkable claim, so there is nothing to export.",
                                        hint="Ask about a relation, e.g. 'Does APT28 use Zebrocy?'")
        bundle = svc.export_stix(ids)
        return {"verification_ids": ids, "bundle": bundle}
    return _call(run)


@router.post("/export/brief")
def export_brief(investigation: Dict[str, Any]):
    from src.analyst.brief import build_brief

    if not isinstance(investigation, dict) or "status" not in investigation:
        return JSONResponse(status_code=400, content={"code": "INVALID_INPUT", "message": "Body must be an /api/investigate result", "hint": "Run an investigation first."})
    md = build_brief(investigation)
    return _download(md, _safe_filename(f"brief_{investigation.get('query', '')}", "md"), "text/markdown; charset=utf-8")


# ------------------------------------------------------------------ analyst tools
@router.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    """Turn an uploaded PDF / TXT / MD / HTML file into plain text (same validation as ingest)."""
    from src.core.uploads import extract_text_from_upload

    data = await file.read()

    def run():
        s = _svc().settings
        text, warnings = extract_text_from_upload(data, file.filename or "upload.txt", s.max_upload_bytes, s.max_upload_pages)
        return {"filename": file.filename, "chars": len(text), "text": text, "warnings": warnings}
    return _call(run)


@router.post("/iocs")
def iocs(req: TextRequest, format: str = "json"):  # noqa: A002
    from src.analyst import ioc

    res = ioc.extract_iocs(req.text, include_reference_domains=req.include_reference_domains)
    if format == "csv":
        return _download(ioc.to_csv(res), _safe_filename(f"iocs_{req.title}", "csv"), "text/csv; charset=utf-8")
    if format == "stix":
        return ioc.to_stix_bundle(res, report_title=req.title)
    return res.to_dict()


@router.post("/factcheck")
def factcheck(req: TextRequest):
    from src.analyst.factcheck import factcheck_report

    return _call(lambda: factcheck_report(_svc(), req.text, title=req.title, publisher=req.publisher, pub_date=req.pub_date))


@router.post("/batch")
def batch(req: BatchRequest, format: str = "json"):  # noqa: A002
    from src.analyst import batch as b

    def run():
        res = b.run_batch(_svc(), req.text)
        if format == "csv":
            return _download(b.to_csv(res), "batch_verification.csv", "text/csv; charset=utf-8")
        return res
    return _call(run)


def _navigator_for(svc, names: List[str], label: str, source_label: str):
    from src.analyst.navigator import build_layer, collect_techniques, tactic_map

    techniques = collect_techniques(svc.retriever.graph.assertions, names)
    registry = getattr(svc, "_registry_cache", None)
    if registry is None:
        registry = json.loads(svc.settings.registry_path.read_text(encoding="utf-8"))
        svc._registry_cache = registry
    return techniques, build_layer(label, techniques, tactic_map(registry), source_label=source_label)


@router.get("/navigator/{name}")
def navigator(name: str, download: bool = False):
    def run():
        svc = _svc()
        info = svc.get_entity(name)
        names = [info.canonical_name, name] + list(info.aliases)
        techniques, layer = _navigator_for(svc, names, info.canonical_name, "CyberRAG corpus")
        if download:
            return _download(json.dumps(layer, indent=2), _safe_filename(f"{info.canonical_name}_attack_layer", "json"), "application/json")
        return {"entity": info.canonical_name, "technique_count": len(techniques), "techniques": techniques, "layer": layer}
    return _call(run)


class ReportLayerRequest(BaseModel):
    title: str = "Analyst report"
    techniques: List[str]


@router.post("/navigator/report")
def navigator_report(req: ReportLayerRequest):
    """Layer from the technique list returned by /factcheck (one evidence point each)."""
    from src.analyst.navigator import build_layer, technique_id

    rows, seen = [], set()
    for t in req.techniques[:500]:
        tid = technique_id(t)
        if tid and tid not in seen:
            seen.add(tid)
            rows.append({"technique_id": tid, "name": t.split(" - ", 1)[-1], "count": 1, "reports": [], "example": ""})
    layer = build_layer(req.title, rows, source_label="uploaded report",
                        note="Techniques were extracted automatically from the uploaded report; review before use.")
    return _download(json.dumps(layer, indent=2), _safe_filename(f"{req.title}_attack_layer", "json"), "application/json")


# ------------------------------------------------------------------ time machine
@router.get("/timetravel")
def timetravel(subject: str, relation: str, object: str, polarity: str = "positive"):  # noqa: A002
    """Verdict today plus the evidence dates at which it can change (slider stops).

    Each stop is then fetched with ``POST /verify_claim`` and ``as_of_date``.
    """
    from src.analyst.timemachine import checkpoints

    def run():
        v = _svc().verify_claim(subject, relation, object, polarity=polarity, use_llm_summary=False)
        stops = checkpoints(v)
        return {
            "claim": v.claim.model_dump(), "polarity": v.polarity, "today": {
                "status": v.status, "support_tier": v.support_tier, "supporting": len(v.supporting),
                "contradicting": len(v.contradicting), "conflict": v.conflict.model_dump(),
            },
            "checkpoints": stops,
            "undated_evidence": sum(1 for e in v.supporting + v.contradicting if not e.pub_date),
        }
    return _call(run)


# ------------------------------------------------------------------ connection finder & comparison
class ConnectRequest(BaseModel):
    a: str
    b: str
    max_hops: int = 4


class CompareRequest(BaseModel):
    names: List[str]


@router.get("/entities")
def entities():
    from src.analyst.graphtools import list_entities

    def run():
        svc = _svc()
        cached = getattr(svc, "_entity_list_cache", None)
        if cached is None:
            cached = list_entities(svc)
            svc._entity_list_cache = cached
        return {"entities": cached}
    return _call(run)


@router.post("/connect")
def connect(req: ConnectRequest):
    from src.analyst.graphtools import find_connections

    return _call(lambda: find_connections(_svc(), req.a, req.b, max_hops=req.max_hops))


@router.post("/compare")
def compare(req: CompareRequest, format: str = "json"):  # noqa: A002
    from src.analyst.graphtools import compare_entities, comparison_csv, overlap_layer
    from src.analyst.navigator import tactic_map

    def run():
        svc = _svc()
        cmp = compare_entities(svc, req.names)
        label = "_vs_".join(e["name"] for e in cmp["entities"])
        if format == "csv":
            return _download(comparison_csv(cmp), _safe_filename(f"compare_{label}", "csv"), "text/csv; charset=utf-8")
        if format == "layer":
            registry = getattr(svc, "_registry_cache", None)
            if registry is None:
                registry = json.loads(svc.settings.registry_path.read_text(encoding="utf-8"))
                svc._registry_cache = registry
            layer = overlap_layer(cmp, tactic_map(registry))
            return _download(json.dumps(layer, indent=2), _safe_filename(f"{label}_attack_overlap", "json"), "application/json")
        return cmp
    return _call(run)


# ------------------------------------------------------------------ entity questions
@router.post("/ask")
def ask(req: QuestionRequest):
    """'What does APT28 target?' / 'Who uses Cobalt Strike?': graph candidates, each NLI-verified."""
    from src.analyst.ask import answer

    return _call(lambda: answer(_svc(), req.question))
