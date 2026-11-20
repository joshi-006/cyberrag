"""/api/check_claims (watchlist re-check) with a stand-in agent, so no models are loaded."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

import demo.server as server


class _Agent:
    def run(self, q):
        if "boom" in q:
            raise RuntimeError("bad claim")
        return SimpleNamespace(status="VERIFIED" if "Zebrocy" in q else "INSUFFICIENT_EVIDENCE",
                               supporting_evidence=[1, 2], contradicting_evidence=[], source_diversity=1, confidence_score=0.8)


def test_check_claims(monkeypatch):
    monkeypatch.setattr(server, "PIPELINE", object())
    monkeypatch.setattr(server, "AGENT", _Agent())
    c = TestClient(server.app)  # no context manager: startup (model loading) does not run
    r = c.post("/api/check_claims", json={"queries": ["Does APT28 use Zebrocy?", "Does X use boom?", "  ", "Does Y use Z?"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["status"] for x in res] == ["VERIFIED", "ERROR", "INSUFFICIENT_EVIDENCE"]
    assert res[0]["supporting"] == 2 and res[0]["score"] == 0.8
    assert c.post("/api/check_claims", json={"queries": []}).status_code == 400
    many = c.post("/api/check_claims", json={"queries": [f"Does A{i} use B?" for i in range(30)]}).json()["results"]
    assert len(many) == 20
