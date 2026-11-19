"""Unit tests for src/analyst: IOC extraction, ATT&CK layers, batch parsing, briefs, fact-check bucketing.

These run without models; the API tests in test_api_v1.py cover the real service.
"""
import json
from types import SimpleNamespace

import pytest

from src.analyst import batch, brief, factcheck, ioc, navigator


# ------------------------------------------------------------------ IOC
SAMPLE = (
    "C2 at hxxps://evil-update[.]com/gate.php and evil-update[.]com, 185.220.101[.]4 and 185.220.101.4 again. "
    "Internal host 10.0.0.5, loopback 127.0.0.1. Dropper invoice.exe (MD5 44d88612fea8a8f36de82e1278abb02f), "
    "SHA1 3395856ce81f2b7382dee72602f798b642f14140, "
    "SHA256 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f. Mail admin[at]bad-domain.ru. "
    "Exploits cve-2021-44228. See https://attack.mitre.org/techniques/T1059 and microsoft.com, e.g. version 1.2.3.4.5"
)


def _vals(res, kind):
    return [i.value for i in res.iocs if i.type == kind]


def test_refang_variants():
    assert ioc.refang("hxxp://a[.]b(.)c{.}d[dot]e") == "http://a.b.c.d.e"
    assert ioc.refang("x[at]y[.]com") == "x@y.com"
    assert ioc.refang("HXXPS[://]host") == "https://host"


def test_extracts_each_type_once_with_counts():
    res = ioc.extract_iocs(SAMPLE)
    assert _vals(res, "ipv4") == ["185.220.101.4"]
    assert [i.count for i in res.iocs if i.type == "ipv4"] == [2]
    assert _vals(res, "url") == ["https://evil-update.com/gate.php"]
    assert "evil-update.com" in _vals(res, "domain")
    assert _vals(res, "email") == ["admin@bad-domain.ru"]
    assert _vals(res, "md5") == ["44d88612fea8a8f36de82e1278abb02f"]
    assert _vals(res, "sha1") == ["3395856ce81f2b7382dee72602f798b642f14140"]
    assert _vals(res, "sha256") == ["275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"]
    assert _vals(res, "cve") == ["CVE-2021-44228"]


def test_exclusions_are_reported_with_reasons():
    res = ioc.extract_iocs(SAMPLE)
    excluded = {(e["type"], e["value"]): e["reason"] for e in res.excluded}
    assert excluded[("ipv4", "10.0.0.5")] == "private address range"
    assert ("ipv4", "127.0.0.1") in excluded
    assert excluded[("domain", "invoice.exe")] == "file name, not a domain"
    assert excluded[("domain", "microsoft.com")] == "well-known reference domain"
    assert any(k[0] == "url" and "attack.mitre.org" in k[1] for k in excluded)
    all_values = {i.value for i in res.iocs}
    assert "microsoft.com" not in all_values and "invoice.exe" not in all_values
    assert not any(v.startswith("1.2.3.4") for v in all_values), "version strings must not become IPs"


def test_hashes_do_not_double_count():
    # a SHA-256 contains 32/40-char hex runs; they must not also show up as MD5/SHA-1
    res = ioc.extract_iocs("hash 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f end")
    assert [i.type for i in res.iocs] == ["sha256"]


def test_reference_domains_can_be_included():
    res = ioc.extract_iocs("see microsoft.com", include_reference_domains=True)
    assert _vals(res, "domain") == ["microsoft.com"]


def test_defang_is_safe_to_paste():
    assert ioc.defang("https://a.b.com/x.php") == "hxxps[://]a[.]b[.]com/x.php"
    assert ioc.defang("1.2.3.4") == "1[.]2[.]3[.]4"
    assert ioc.defang("a@b.ru") == "a@b[.]ru"


def test_deterministic_and_empty():
    assert ioc.extract_iocs(SAMPLE).to_dict() == ioc.extract_iocs(SAMPLE).to_dict()
    assert ioc.extract_iocs("no indicators here").to_dict()["total"] == 0
    with pytest.raises(TypeError):
        ioc.extract_iocs(None)


def test_csv_export_has_header_and_rows():
    out = ioc.to_csv(ioc.extract_iocs(SAMPLE)).splitlines()
    assert out[0] == "type,value,defanged,count,context"
    assert len(out) == 1 + len(ioc.extract_iocs(SAMPLE).iocs)


def test_stix_bundle_is_valid_and_deterministic_ids():
    from stix2validator import validate_instance

    res = ioc.extract_iocs(SAMPLE)
    b1, b2 = ioc.to_stix_bundle(res, "t"), ioc.to_stix_bundle(res, "t")
    assert validate_instance(b1).is_valid
    types = sorted(o["type"] for o in b1["objects"])
    assert types.count("vulnerability") == 1
    assert types.count("indicator") == len(res.iocs) - 1
    assert sorted(o["id"] for o in b1["objects"]) == sorted(o["id"] for o in b2["objects"])
    assert ioc.to_stix_bundle(ioc.extract_iocs("nothing"))["objects"] == []


def test_stix_pattern_escapes_quotes():
    i = ioc.IOC(type="url", value="http://x.com/a'b", defanged="")
    assert "\\'" in ioc._stix_pattern(i)


# ------------------------------------------------------------------ ATT&CK Navigator
ASSERTIONS = [
    {"predicate": "USES_TECHNIQUE", "subject": "Leviathan", "object": "T1021 - Remote Services", "relation_id": "r1", "report_id": "R1", "evidence_sentence": "x"},
    {"predicate": "USES_TECHNIQUE", "subject": "leviathan", "object": "T1021 - Remote Services", "relation_id": "r2", "report_id": "R2"},
    {"predicate": "USES_TECHNIQUE", "subject": "Leviathan", "object": "T1059.001 - PowerShell", "relation_id": "r3", "report_id": "R1"},
    {"predicate": "USES_TECHNIQUE", "subject": "Leviathan", "object": "T1105 - Ingress", "relation_id": "r4", "polarity": "negative"},
    {"predicate": "USES", "subject": "Leviathan", "object": "T1000 - Not a technique edge", "relation_id": "r5"},
    {"predicate": "USES_TECHNIQUE", "subject": "Other", "object": "T1566 - Phishing", "relation_id": "r6"},
    {"predicate": "USES_TECHNIQUE", "subject": "Leviathan", "object": "no id here", "relation_id": "r7"},
]


def test_technique_id_parsing():
    assert navigator.technique_id("T1059.007 - JavaScript") == "T1059.007"
    assert navigator.technique_id("T1021") == "T1021"
    assert navigator.technique_id("Phishing") is None


def test_collect_techniques_filters_and_counts():
    rows = navigator.collect_techniques(ASSERTIONS, ["Leviathan"])
    assert [r["technique_id"] for r in rows] == ["T1021", "T1059.001"]
    assert rows[0]["count"] == 2 and rows[0]["reports"] == ["R1", "R2"]


def test_build_layer_shape():
    rows = navigator.collect_techniques(ASSERTIONS, ["Leviathan"])
    layer = navigator.build_layer("Leviathan", rows, {"T1021": "lateral-movement"})
    assert layer["domain"] == "enterprise-attack" and layer["versions"]["layer"] == "4.5"
    t = {e["techniqueID"]: e for e in layer["techniques"]}
    assert t["T1021"]["score"] == 2 and "lateral-movement" in t["T1021"]["comment"]
    assert t["T1059.001"]["showSubtechniques"] is True
    assert layer["gradient"]["maxValue"] == 2
    assert "not human reviewed" in layer["description"]
    json.dumps(layer)  # serialisable


def test_empty_layer_is_valid():
    layer = navigator.build_layer("X", [])
    assert layer["techniques"] == [] and layer["gradient"]["maxValue"] == 1


def test_tactic_map_from_registry():
    reg = {"techniques": {"T1": {"tactic": "execution"}, "T2": {"tactic": ["a", "b"]}, "T3": {}}}
    assert navigator.tactic_map(reg) == {"T1": "execution", "T2": "a, b"}


# ------------------------------------------------------------------ batch parsing
@pytest.mark.parametrize("line,expected", [
    ("APT28 | USES | Zebrocy", ("claim", "APT28", "USES", "Zebrocy", "positive")),
    ("APT28,uses,Zebrocy", ("claim", "APT28", "USES", "Zebrocy", "positive")),
    ("APT28;targets;NATO", ("claim", "APT28", "TARGETS", "NATO", "positive")),
    ("APT28\tUSES TECHNIQUE\tT1566", ("claim", "APT28", "USES_TECHNIQUE", "T1566", "positive")),
    ("Emotet | NOT USES | TrickBot", ("claim", "Emotet", "USES", "TrickBot", "negative")),
    ('"Lazarus","exploit","CVE-2021-44228"', ("claim", "Lazarus", "EXPLOITS", "CVE-2021-44228", "positive")),
])
def test_parse_structured(line, expected):
    p = batch.parse_line(line)
    assert (p["kind"], p["subject"], p["relation"], p["object"], p["polarity"]) == expected


def test_parse_skips_and_questions():
    assert batch.parse_line("") is None
    assert batch.parse_line("# comment") is None
    assert batch.parse_line("subject,relation,object") is None
    assert batch.parse_line("Does APT29 use Cobalt Strike?")["kind"] == "question"
    # commas inside a sentence are still a question, not a malformed claim
    assert batch.parse_line("Is APT29, the SVR group, using WellMess?")["kind"] == "question"


def test_parse_unknown_relation_is_reported():
    p = batch.parse_line("foo | FLIES | bar")
    assert p["kind"] == "invalid" and "FLIES" in p["error"]


class _FakeVerification:
    def __init__(self, s, r, o, polarity):
        self._d = {
            "claim": {"subject": s, "relation": r, "object": o}, "polarity": polarity,
            "status": "SUPPORTED" if s == "APT28" else "INSUFFICIENT_EVIDENCE",
            "support_tier": "SINGLE_SOURCE" if s == "APT28" else "NONE",
            "supporting": [{"publisher": "Vendor"}] if s == "APT28" else [], "contradicting": [],
            "hedged_only": False, "conflict": {"type": "NO_CONFLICT"}, "verification_id": f"v_{s}",
        }

    def model_dump(self):
        return self._d


class _FakeService:
    def verify_claim(self, s, r, o, polarity="positive", use_llm_summary=True):
        from src.core.errors import InvalidInputError

        if s == "boom":
            raise InvalidInputError("bad entity")
        return _FakeVerification(s, r, o, polarity)

    def verify_question(self, q):
        return {"overall_status": "INSUFFICIENT_EVIDENCE", "results": []}


def test_run_batch_mixed_rows_and_csv():
    text = "APT28 | USES | Zebrocy\nboom | USES | x\nfoo | FLIES | bar\nWhat is up?\nX | NOT USES | Y"
    res = batch.run_batch(_FakeService(), text)
    statuses = [r["status"] for r in res["rows"]]
    assert statuses == ["SUPPORTED", "ERROR", "ERROR", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"]
    assert res["rows"][0]["publishers"] == ["Vendor"]
    assert res["rows"][4]["claim"] == "X NOT USES Y"
    assert res["summary"] == {"SUPPORTED": 1, "ERROR": 2, "INSUFFICIENT_EVIDENCE": 2}
    csv_lines = batch.to_csv(res).splitlines()
    assert csv_lines[0].startswith("input,claim,status") and len(csv_lines) == 6


def test_run_batch_caps_lines():
    res = batch.run_batch(_FakeService(), "\n".join(f"APT28 | USES | m{i}" for i in range(40)))
    assert res["lines_read"] == batch.MAX_LINES and res["truncated"] is True


# ------------------------------------------------------------------ brief
INV = {
    "query": "Does APT28 use Zebrocy?", "status": "TEMPORAL_SUPERSEDENCE", "consensus_evidence_score": 0.8,
    "source_diversity": 2, "matched_entities": [{"name": "APT28"}, {"name": "Zebrocy"}],
    "supporting_evidence": [{"source": "Vendor A", "publication_date": "2020-01-01", "text": "APT28 uses Zebrocy.", "p_entail": 0.97, "segment_id": "R1::p0::s0"}],
    "contradicting_evidence": [{"source": "Vendor B", "publication_date": "2021-01-01",
                                "text": "# Injected heading\n[click](http://evil) <script>x</script>", "p_contra": 0.9, "segment_id": "R2::p1::s0"}],
    "conflict_resolution": {"status": "TEMPORAL_SUPERSEDENCE", "delta_days": 366, "threshold": 90, "reason": "Later report."},
    "execution_trace": [{"step_index": 1, "state": "QUERY_ANALYSIS", "action": "Plan"}],
}


def test_brief_contains_verdict_evidence_and_limits():
    md = brief.build_brief(INV, generated_at="2024-01-01 00:00 UTC")
    assert "# CTI verification brief: Does APT28 use Zebrocy?" in md
    assert "**TEMPORAL\\_SUPERSEDENCE**" in md
    assert "366 days between reports" in md
    assert "Vendor A" in md and "Vendor B" in md
    assert "Passage `R1::p0::s0`" in md
    assert "## Limits" in md


def test_brief_neutralises_markdown_injection():
    md = brief.build_brief(INV)
    assert "\n# Injected" not in md
    # Markdown control characters are backslash-escaped, so they render as literal text
    assert "\\[click\\](http://evil)" in md
    assert "\\<script\\>" in md
    assert "[click](http://evil)" not in md and "<script>" not in md.replace("\\<script\\>", "")


def test_brief_handles_empty_investigation():
    md = brief.build_brief({"status": "INSUFFICIENT_EVIDENCE", "query": "q"})
    assert "No supporting evidence" in md and "No contradicting evidence" in md


# ------------------------------------------------------------------ fact-check
def test_bucket_mapping():
    assert factcheck.bucket_for("SUPPORTED") == "CORROBORATED"
    assert factcheck.bucket_for("CONTRADICTED") == "CONTRADICTED_BY_CORPUS"
    assert factcheck.bucket_for("CONFLICTING_EVIDENCE") == "DISPUTED"
    assert factcheck.bucket_for("INSUFFICIENT_EVIDENCE") == "NEW_INTEL"
    assert factcheck.bucket_for("???") == "ERROR"


def test_dedupe_claims_keeps_best_and_counts():
    rels = [
        {"subject": "A", "relation": "USES", "object": "B", "confidence": 0.5, "evidence_text": "low"},
        {"subject": "a", "relation": "USES", "object": "b", "confidence": 0.9, "evidence_text": "high"},
        {"subject": "A", "relation": "USES", "object": "A", "confidence": 1.0},  # self-loop dropped
        {"subject": "C", "relation": "TARGETS", "object": "D", "confidence": 0.99},
        {"subject": "A", "relation": "USES", "object": "B", "polarity": "negative", "confidence": 0.7},
    ]
    rows = factcheck.dedupe_claims(rels)
    assert len(rows) == 3
    ab = rows[0]
    assert ab["occurrences"] == 2 and ab["evidence_text"] == "high"


class _FCService:
    settings = SimpleNamespace(max_upload_bytes=1_000_000)

    class _Ext:
        def extract(self, text, **kw):
            return {
                "segments": [{"id": 1}, {"id": 2}],
                "entities": [{"name": "APT28"}],
                "relations": [
                    {"subject": "APT28", "relation": "USES", "object": "Zebrocy", "confidence": 0.9, "evidence_text": "s1"},
                    {"subject": "Newbie", "relation": "USES", "object": "Tool", "confidence": 0.8, "evidence_text": "s2"},
                    {"subject": "boom", "relation": "USES", "object": "x", "confidence": 0.8, "evidence_text": "s3"},
                    {"subject": "APT28", "relation": "USES_TECHNIQUE", "object": "T1566 - Phishing", "confidence": 0.8, "evidence_text": "s4"},
                ],
            }

    def _extract(self):
        return self._Ext()

    def verify_claim(self, s, r, o, polarity="positive", use_llm_summary=True):
        from src.core.errors import InvalidInputError
        from src.core.models import ClaimSpec, ClaimVerification, EvidenceItem

        if s == "boom":
            raise InvalidInputError("bad")
        status = "SUPPORTED" if (s, o) == ("APT28", "Zebrocy") else "INSUFFICIENT_EVIDENCE"
        sup = [EvidenceItem(evidence_id="e1", publisher="V", pub_date="2020-01-01", untrusted_text="APT28 uses Zebrocy")] if status == "SUPPORTED" else []
        return ClaimVerification(
            verification_id=f"v_{s}_{o}", claim=ClaimSpec(subject=s, relation=r, object=o), status=status,
            support_tier="SINGLE_SOURCE" if sup else "NONE", polarity=polarity, supporting=sup, trace_id="t",
        )


def test_factcheck_report_buckets_and_techniques():
    out = factcheck.factcheck_report(_FCService(), "APT28 uses Zebrocy. Newbie uses Tool.", title="T")
    by = {(c["subject"], c["object"]): c for c in out["claims"]}
    assert by[("APT28", "Zebrocy")]["bucket"] == "CORROBORATED"
    assert by[("APT28", "Zebrocy")]["supporting"][0]["publisher"] == "V"
    assert by[("Newbie", "Tool")]["bucket"] == "NEW_INTEL"
    assert by[("boom", "x")]["bucket"] == "ERROR" and by[("boom", "x")]["error"] == "bad"
    assert out["summary"]["CORROBORATED"] == 1 and out["summary"]["ERROR"] == 1
    assert out["techniques"] == ["T1566 - Phishing"]
    assert out["claims_checked"] == 4 and out["truncated"] is False
    assert "one third" in out["note"]


def test_factcheck_rejects_empty_text():
    from src.core.errors import CTIError

    with pytest.raises(CTIError):
        factcheck.factcheck_report(_FCService(), "   ")


# ------------------------------------------------------------------ time machine
from src.analyst import timemachine  # noqa: E402


def _v(status="SUPPORTED", tier="SINGLE_SOURCE", hedged=False, conflict="NO_CONFLICT", sup_dates=(), con_dates=()):
    from src.core.models import ClaimSpec, ClaimVerification, Conflict, EvidenceItem

    mk = lambda d, i: EvidenceItem(evidence_id=f"e{i}_{d}", publisher="P", pub_date=d, untrusted_text="t")  # noqa: E731
    return ClaimVerification(
        verification_id="v", claim=ClaimSpec(subject="A", relation="USES", object="B"), status=status,
        support_tier=tier, polarity="positive", hedged_only=hedged, conflict=Conflict(type=conflict),
        supporting=[mk(d, i) for i, d in enumerate(sup_dates)], contradicting=[mk(d, 100 + i) for i, d in enumerate(con_dates)],
        trace_id="t",
    )


def test_checkpoints_are_sorted_distinct_dates():
    v = _v(sup_dates=("2021-01-02", "2020-05-01", "2020-05-01T10:00"), con_dates=("2021-06-01",))
    assert timemachine.checkpoints(v) == ["2020-05-01", "2021-01-02", "2021-06-01"]


def test_checkpoints_keep_every_contradiction_when_capped():
    sup = tuple(f"2020-{m:02d}-01" for m in range(1, 13)) + tuple(f"2021-{m:02d}-01" for m in range(1, 13))
    con = ("2020-07-15", "2021-03-15")
    stops = timemachine.checkpoints(_v(sup_dates=sup, con_dates=con), limit=8)
    assert len(stops) <= 8 and set(con) <= set(stops)
    assert stops[0] == "2020-01-01" and stops[-1] == "2021-12-01" and stops == sorted(stops)


def test_checkpoints_ignore_undated():
    assert timemachine.checkpoints(_v(sup_dates=("",))) == []


# ------------------------------------------------------------------ connection finder & comparison
from src.analyst import graphtools  # noqa: E402


class _GraphService:
    """Tiny graph: A -USES-> T <-USES- B -TARGETS-> X ; A -TARGETS-> HUB <-TARGETS- (many)."""

    def __init__(self):
        nodes = [("ThreatActor::A", "A"), ("ThreatActor::B", "B"), ("Tool::T", "T"), ("Organization::X", "X"),
                 ("Organization::HUB", "HUB"), ("ThreatActor::C", "C"), ("Technique::T1566 - Phishing", "T1566 - Phishing"),
                 ("Malware::Lonely", "Lonely")]
        nodes += [(f"ThreatActor::H{i}", f"H{i}") for i in range(8)]
        links = [
            ("ThreatActor::A", "Tool::T", "USES"), ("ThreatActor::B", "Tool::T", "USES"),
            ("ThreatActor::B", "Organization::X", "TARGETS"), ("ThreatActor::A", "Organization::HUB", "TARGETS"),
            ("ThreatActor::C", "Organization::HUB", "TARGETS"), ("ThreatActor::A", "Technique::T1566 - Phishing", "USES_TECHNIQUE"),
            ("ThreatActor::B", "Technique::T1566 - Phishing", "USES_TECHNIQUE"),
        ] + [(f"ThreatActor::H{i}", "Organization::HUB", "TARGETS") for i in range(8)]
        self.nodes = {nid: {"id": nid, "name": name, "category": nid.split("::")[0], "mention_count": 3} for nid, name in nodes}
        by = {}
        for s, t, r in links:
            l = {"source": s, "target": t, "relation_type": r, "evidence_text": f"{s} {r} {t}", "source_publisher": "V",
                 "publication_date": "2021-01-01", "report_id": "R", "sources": ["V"], "evidence_count": 1}
            by.setdefault(s, []).append(l)
            by.setdefault(t, []).append(l)
        self.idx = {"nodes": self.nodes, "links": by}
        self.aliases = SimpleNamespace(resolve=lambda name: name)
        self.verified = []

    def _graph(self):
        return self.idx

    def _find_node(self, name):
        return next((n for n in self.nodes.values() if n["name"].lower() == name.lower()), None)

    def verify_claim(self, s, r, o, use_llm_summary=True):
        self.verified.append((s, r, o))
        return _v(status="SUPPORTED" if s == "B" else "INSUFFICIENT_EVIDENCE", tier="SINGLE_SOURCE" if s == "B" else "NONE")

    def get_entity(self, name):
        return SimpleNamespace(mitre_id=None, aliases=[], first_seen=None, last_seen=None)


def test_connections_prefer_specific_links_over_hubs():
    svc = _GraphService()
    r = graphtools.find_connections(svc, "A", "X", max_paths=3)
    first = r["paths"][0]
    assert [n["name"] for n in first["nodes"]] == ["A", "T", "B", "X"] or [n["name"] for n in first["nodes"]] == ["A", "T1566 - Phishing", "B", "X"]
    assert all("HUB" not in [n["name"] for n in p["nodes"]] for p in r["paths"][:2])
    hop = first["hops"][0]
    assert hop["subject"] == "A" and hop["forward"] is True and hop["publisher"] == "V"
    back = first["hops"][1]
    assert back["subject"] == "B" and back["forward"] is False  # B USES T, walked backwards
    assert first["confirmed_hops"] == sum(1 for h in first["hops"] if h["subject"] == "B")


def test_connections_verify_each_hop_once():
    svc = _GraphService()
    graphtools.find_connections(svc, "A", "X", max_paths=3)
    assert len(svc.verified) == len(set(svc.verified))


def test_connections_no_path_and_errors():
    from src.core.errors import EntityNotFoundError, InvalidInputError

    svc = _GraphService()
    assert graphtools.find_connections(svc, "A", "Lonely")["paths"] == []
    with pytest.raises(EntityNotFoundError):
        graphtools.find_connections(svc, "A", "Nobody")
    with pytest.raises(InvalidInputError):
        graphtools.find_connections(svc, "A", "a")
    r = graphtools.find_connections(svc, "A", "X", max_hops=1)
    assert r["paths"] == []  # shortest is 3 hops


def test_compare_shared_and_unique():
    svc = _GraphService()
    c = graphtools.compare_entities(svc, ["A", "B"])
    groups = {g["key"]: g for g in c["groups"]}
    uses = {it["name"]: it["shared_by"] for it in groups["uses"]["items"]}
    assert uses == {"T": ["A", "B"]}
    tech = groups["techniques"]["items"][0]
    assert tech["name"] == "T1566 - Phishing" and tech["shared_by"] == ["A", "B"]
    targets = {it["name"]: it["shared_by"] for it in groups["targets"]["items"]}
    assert targets == {"HUB": ["A"], "X": ["B"]}
    assert c["shared_all"] == 2 and c["pairs"][0]["shared"] == 2
    assert groups["uses"]["items"][0]["evidence"]["A"][0]["publisher"] == "V"


def test_compare_validation_and_exports():
    from src.core.errors import InvalidInputError

    svc = _GraphService()
    with pytest.raises(InvalidInputError):
        graphtools.compare_entities(svc, ["A"])
    with pytest.raises(InvalidInputError):
        graphtools.compare_entities(svc, ["A", "a"])
    c = graphtools.compare_entities(svc, ["A", "B", "C"])
    csv_lines = graphtools.comparison_csv(c).splitlines()
    assert csv_lines[0] == "group,item,category,A,B,C"
    layer = graphtools.overlap_layer(c)
    assert [(t["techniqueID"], t["score"]) for t in layer["techniques"]] == [("T1566", 2)]


def test_list_entities_only_linked():
    names = {e["name"] for e in graphtools.list_entities(_GraphService())}
    assert "Lonely" not in names and {"A", "B", "HUB"} <= names


# ------------------------------------------------------------------ entity questions
from src.analyst import ask  # noqa: E402


@pytest.mark.parametrize("q,entity,expected", [
    ("What does APT28 target?", "APT28", ("TARGETS", False, None)),
    ("Who uses Cobalt Strike?", "Cobalt Strike", ("USES", True, None)),
    ("Which malware does Emotet use?", "Emotet", ("USES", False, "Malware")),
    ("What techniques does Leviathan use?", "Leviathan", ("USES_TECHNIQUE", False, "Technique")),
    ("Which groups target government?", "government", ("TARGETS", True, "ThreatActor")),
    ("Which CVEs does Lazarus exploit?", "Lazarus", ("EXPLOITS", False, "CVE")),
    ("What does TrickBot deliver?", "TrickBot", ("USES", False, None)),
    ("Tell me about Winnti", "Winnti", (None, False, None)),
])
def test_parse_entity_question(q, entity, expected):
    p = ask.parse(q, entity)
    assert (p["relation"], p["inbound"], p["category"]) == expected


def test_is_entity_question():
    assert ask.is_entity_question("What does APT28 target?")
    assert ask.is_entity_question("who uses Cobalt Strike")
    assert not ask.is_entity_question("Does APT28 use Zebrocy?")


class _AskService(_GraphService):
    def __init__(self, entities):
        super().__init__()
        self.planner = SimpleNamespace(plan=lambda q: (SimpleNamespace(entities=entities), []))

    def verify_claim(self, s, r, o, use_llm_summary=True):
        return _v(status="SUPPORTED" if o == "X" else "INSUFFICIENT_EVIDENCE", tier="SINGLE_SOURCE" if o == "X" else "NONE",
                  sup_dates=("2021-01-01",) if o == "X" else ())


def test_answer_outbound_and_order():
    r = ask.answer(_AskService(["B"]), "What does B target?")
    assert r["relation"] == "TARGETS" and [a["object"] for a in r["answers"]] == ["X"]
    assert r["answers"][0]["status"] == "SUPPORTED" and r["confirmed"] == 1
    assert r["answers"][0]["quote"]["publisher"] == "P"


def test_answer_inbound_and_profile():
    r = ask.answer(_AskService(["T"]), "Who uses T?")
    assert sorted(a["subject"] for a in r["answers"]) == ["A", "B"] and all(a["object"] == "T" for a in r["answers"])
    prof = ask.answer(_AskService(["B"]), "Tell me about B")
    assert {(a["relation"], a["object"]) for a in prof["answers"]} == {("USES", "T"), ("TARGETS", "X"), ("USES_TECHNIQUE", "T1566 - Phishing")}
    assert prof["answers"][0]["object"] == "X"  # confirmed answers first


def test_answer_errors():
    from src.core.errors import EntityNotFoundError, InvalidInputError

    with pytest.raises(InvalidInputError):
        ask.answer(_AskService([]), "What does it do?")
    with pytest.raises(EntityNotFoundError):
        ask.answer(_AskService(["Nobody"]), "What does Nobody target?")


def test_resolve_prefers_exact_linked_name():
    svc = _GraphService()
    svc._find_node = lambda name: svc.nodes["Malware::Lonely"]  # alias resolver points somewhere unlinked
    assert graphtools.resolve_node(svc, "hub")["id"] == "Organization::HUB"
