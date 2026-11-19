"""End-to-end tests for the /api/v1 router with the real VerificationService (models loaded once)."""
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from demo import api_v1
from src.core.bootstrap import build_service
from src.core.config import Settings


@pytest.fixture(scope="module")
def client():
    svc = build_service(Settings())
    app = FastAPI()
    app.include_router(api_v1.router)
    api_v1.set_service_provider(lambda: svc)
    with TestClient(app) as c:
        yield c
    api_v1.set_service_provider(None)


def test_stats(client):
    r = client.get("/api/v1/stats")
    assert r.status_code == 200
    d = r.json()
    assert d["segments"] == 5991 and d["assertions"] == 374 and d["verifier_ready"] is True


def test_verify_claim_and_trace(client):
    r = client.post("/api/v1/verify_claim", json={"subject": "APT28", "relation": "USES", "object": "Zebrocy"})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "SUPPORTED" and d["support_tier"] == "SINGLE_SOURCE"
    t = client.get(f"/api/v1/trace/{d['verification_id']}")
    assert t.status_code == 200


def test_bad_relation_is_400(client):
    r = client.post("/api/v1/verify_claim", json={"subject": "APT28", "relation": "FLIES", "object": "Zebrocy"})
    assert r.status_code == 400 and r.json()["code"] in ("UNKNOWN_RELATION", "INVALID_INPUT")


def test_entity_found_and_not_found(client):
    r = client.get("/api/v1/entity/Fancy Bear")
    assert r.status_code == 200 and r.json()["canonical_name"] == "APT28"
    r = client.get("/api/v1/entity/NoSuchThing123")
    assert r.status_code == 404 and r.json()["code"] == "ENTITY_NOT_FOUND"


def test_search_and_timeline(client):
    r = client.post("/api/v1/search", json={"query": "Emotet TrickBot", "top_k": 3})
    assert r.status_code == 200 and len(r.json()["results"]) <= 3
    r = client.get("/api/v1/timeline", params={"subject": "Emotet", "object": "TrickBot", "relation": "USES"})
    assert r.status_code == 200 and r.json()["entries"]


def test_export_stix_from_question_is_valid(client):
    from stix2validator import validate_instance

    r = client.post("/api/v1/export/stix", json={"question": "Does APT28 use Zebrocy?"})
    assert r.status_code == 200
    d = r.json()
    assert d["verification_ids"] and validate_instance(d["bundle"]).is_valid
    assert any(o["type"] == "relationship" for o in d["bundle"]["objects"])


def test_export_stix_without_claim_is_400(client):
    r = client.post("/api/v1/export/stix", json={"question": "hello there"})
    assert r.status_code == 400


def test_export_brief_download(client):
    inv = {"query": "Does APT28 use Zebrocy?", "status": "VERIFIED", "supporting_evidence": [], "contradicting_evidence": []}
    r = client.post("/api/v1/export/brief", json=inv)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "attachment" in r.headers["content-disposition"]
    assert r.text.startswith("# CTI verification brief")
    assert client.post("/api/v1/export/brief", json={"nope": 1}).status_code == 400


def test_iocs_json_csv_stix(client):
    body = {"text": "C2 hxxp://bad[.]top/x and 91.219.236[.]166, CVE-2021-44228", "title": "t"}
    d = client.post("/api/v1/iocs", json=body).json()
    assert {i["type"] for i in d["iocs"]} == {"url", "domain", "ipv4", "cve"}
    csv = client.post("/api/v1/iocs?format=csv", json=body)
    assert csv.headers["content-type"].startswith("text/csv") and csv.text.startswith("type,value")
    stix = client.post("/api/v1/iocs?format=stix", json=body).json()
    assert stix["type"] == "bundle" and len(stix["objects"]) == 4


def test_factcheck_real_report(client):
    text = ("APT28 used the Zebrocy malware in a phishing campaign against NATO members. "
            "APT29 used Cobalt Strike beacons for lateral movement. "
            "Black Basta exploited CVE-2024-1709 in ScreenConnect servers.")
    r = client.post("/api/v1/factcheck", json={"text": text, "title": "digest", "pub_date": "2024-03-01"})
    assert r.status_code == 200
    d = r.json()
    by = {(c["subject"], c["relation"], c["object"]): c for c in d["claims"]}
    assert by[("APT28", "USES", "Zebrocy")]["bucket"] == "CORROBORATED"
    assert by[("Black Basta", "EXPLOITS", "CVE-2024-1709")]["bucket"] == "NEW_INTEL"
    assert sum(d["summary"].values()) == d["claims_checked"]


def test_factcheck_does_not_change_corpus(client):
    before = client.get("/api/v1/stats").json()
    client.post("/api/v1/factcheck", json={"text": "NewActorX used NewToolY against banks."})
    after = client.get("/api/v1/stats").json()
    assert (before["segments"], before["graph_nodes"], before["assertions"]) == (after["segments"], after["graph_nodes"], after["assertions"])


def test_batch_json_and_csv(client):
    text = "APT28 | USES | Zebrocy\nfoo | FLIES | bar\nDoes APT29 use Cobalt Strike?"
    d = client.post("/api/v1/batch", json={"text": text}).json()
    assert [r["status"] for r in d["rows"]] == ["SUPPORTED", "ERROR", "SUPPORTED"]
    csv = client.post("/api/v1/batch?format=csv", json={"text": text})
    assert csv.status_code == 200 and len(csv.text.strip().splitlines()) == 4


def test_navigator_entity_and_download(client):
    d = client.get("/api/v1/navigator/Leviathan").json()
    assert d["technique_count"] > 0 and d["layer"]["domain"] == "enterprise-attack"
    dl = client.get("/api/v1/navigator/Leviathan?download=true")
    assert dl.headers["content-type"].startswith("application/json")
    assert json.loads(dl.text)["techniques"]
    assert client.get("/api/v1/navigator/NoSuchThing123").status_code == 404


def test_navigator_report_layer(client):
    r = client.post("/api/v1/navigator/report", json={"title": "r", "techniques": ["T1566 - Phishing", "T1566 - Phishing", "junk"]})
    layer = json.loads(r.text)
    assert [t["techniqueID"] for t in layer["techniques"]] == ["T1566"]


def test_extract_text_upload(client):
    r = client.post("/api/v1/extract_text", files={"file": ("r.txt", b"APT28 used the Zebrocy malware against NATO members.", "text/plain")})
    assert r.status_code == 200 and "Zebrocy" in r.json()["text"]
    bad = client.post("/api/v1/extract_text", files={"file": ("r.exe", b"MZ\x00\x01", "application/octet-stream")})
    assert bad.status_code in (400, 422)


def test_warming_up_returns_503():
    app = FastAPI()
    app.include_router(api_v1.router)

    def not_ready():
        raise api_v1.ServiceWarmingUp("loading")

    api_v1.set_service_provider(not_ready)
    try:
        with TestClient(app) as c:
            r = c.get("/api/v1/stats")
            assert r.status_code == 503 and r.json()["code"] == "WARMING_UP"
            # IOC extraction needs no models, so it keeps working while the service loads
            assert c.post("/api/v1/iocs", json={"text": "1.1.1.1"}).status_code == 200
    finally:
        api_v1.set_service_provider(None)
