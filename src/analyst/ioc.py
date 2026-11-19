"""Indicator-of-compromise (IOC) extraction from CTI report text.

Finds IPv4 addresses, domains, URLs, e-mail addresses, MD5/SHA-1/SHA-256 hashes
and CVE ids, refanging the usual defanging styles first (``hxxp``, ``[.]``,
``(.)``, ``[:]``, ``[at]``). Values that are almost certainly not indicators are
not silently dropped: they are returned under ``excluded`` with a reason
(private / reserved IP, well-known reference domain, file name that looks like a
domain), so an analyst can see what was filtered and why.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

MAX_TEXT_CHARS = 2_000_000

# ------------------------------------------------------------------ refanging
_REFANG_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bhxxp(s?)", re.I), lambda m: "http" + m.group(1).lower()),
    (re.compile(r"\bfxp\b", re.I), "ftp"),
    (re.compile(r"\[\s*\.\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}|\[dot\]|\(dot\)", re.I), "."),
    (re.compile(r"\[\s*:\s*\]"), ":"),
    (re.compile(r"\[\s*/\s*\]"), "/"),
    (re.compile(r"\[\s*@\s*\]|\[at\]|\(at\)", re.I), "@"),
    (re.compile(r"\[://\]"), "://"),
]


def refang(text: str) -> str:
    for pat, repl in _REFANG_RULES:
        text = pat.sub(repl, text)
    return text


def defang(value: str) -> str:
    """Safe-to-share form: ``http`` -> ``hxxp`` and every dot in the host -> ``[.]``."""
    m = re.match(r"^([a-z]+)://([^/]*)(.*)$", value, flags=re.I)
    if m:
        scheme = re.sub(r"^http", "hxxp", m.group(1), flags=re.I)
        return f"{scheme}[://]{m.group(2).replace('.', '[.]')}{m.group(3)}"
    return value.replace(".", "[.]")


# ------------------------------------------------------------------ patterns
_IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.]*\d)")
_URL = re.compile(r"\b(?:https?|ftp)://[^\s<>\"'`\]\)]+", re.I)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b")
_DOMAIN = re.compile(r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}\b")
_CVE = re.compile(r"\bCVE-(?:19|20)\d{2}-\d{4,7}\b", re.I)
_HASHES = [
    ("sha256", re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")),
    ("sha1", re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{40}(?![0-9A-Fa-f])")),
    ("md5", re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{32}(?![0-9A-Fa-f])")),
]

# Suffixes that make "name.ext" a file name, not a domain.
_FILE_EXTENSIONS = {
    "exe", "dll", "sys", "bat", "cmd", "ps1", "vbs", "js", "jse", "hta", "lnk", "scr", "msi", "jar", "py", "sh",
    "doc", "docx", "docm", "xls", "xlsx", "xlsm", "ppt", "pptx", "pdf", "rtf", "txt", "csv", "json", "xml", "yml",
    "yaml", "ini", "cfg", "conf", "log", "tmp", "dat", "bin", "zip", "rar", "7z", "gz", "tar", "iso", "img", "png",
    "jpg", "jpeg", "gif", "bmp", "svg", "html", "htm", "php", "asp", "aspx", "jsp", "md", "db", "sqlite", "lock",
}
# Well-known reference / vendor domains that appear in reports as citations, not IOCs.
_REFERENCE_DOMAINS = {
    "microsoft.com", "windows.com", "google.com", "github.com", "githubusercontent.com", "mitre.org", "attack.mitre.org",
    "cisa.gov", "us-cert.gov", "nist.gov", "nvd.nist.gov", "virustotal.com", "twitter.com", "x.com", "linkedin.com",
    "youtube.com", "wikipedia.org", "apple.com", "amazon.com", "cloudflare.com", "mozilla.org", "w3.org",
    "mandiant.com", "fireeye.com", "crowdstrike.com", "kaspersky.com", "securelist.com", "symantec.com",
    "broadcom.com", "trendmicro.com", "paloaltonetworks.com", "unit42.paloaltonetworks.com", "zscaler.com",
    "proofpoint.com", "eset.com", "welivesecurity.com", "sophos.com", "talosintelligence.com", "cisco.com",
    "intel471.com", "recordedfuture.com", "checkpoint.com", "mcafee.com", "fortinet.com", "europol.europa.eu",
    "fbi.gov", "ic3.gov", "ncsc.gov.uk", "example.com", "example.org", "example.net",
}
# Common abbreviations / words with a dot that look like domains.
_FALSE_DOMAINS = {"e.g", "i.e", "u.s", "u.k", "a.m", "p.m", "etc.al"}


@dataclass
class IOC:
    type: str
    value: str
    defanged: str
    count: int = 1
    context: str = ""


@dataclass
class IOCResult:
    iocs: List[IOC] = field(default_factory=list)
    excluded: List[Dict[str, str]] = field(default_factory=list)

    def by_type(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in self.iocs:
            out[i.type] = out.get(i.type, 0) + 1
        return out

    def to_dict(self) -> Dict:
        return {
            "total": len(self.iocs),
            "by_type": self.by_type(),
            "iocs": [asdict(i) for i in self.iocs],
            "excluded": self.excluded,
        }


def _context(text: str, start: int, end: int, width: int = 80) -> str:
    a, b = max(0, start - width), min(len(text), end + width)
    snippet = text[a:b].replace("\n", " ").strip()
    return ("…" if a > 0 else "") + snippet + ("…" if b < len(text) else "")


def _registered_domain(host: str) -> str:
    parts = host.lower().strip(".").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()


def _is_reference(host: str) -> bool:
    h = host.lower().strip(".")
    return h in _REFERENCE_DOMAINS or _registered_domain(h) in _REFERENCE_DOMAINS


def _ip_exclusion(ip: str) -> Optional[str]:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "not a valid IP address"
    if addr.is_private:
        return "private address range"
    if addr.is_loopback or addr.is_unspecified or addr.is_reserved or addr.is_multicast or addr.is_link_local:
        return "reserved / non-routable address"
    return None


def extract_iocs(text: str, include_reference_domains: bool = False) -> IOCResult:
    """Extract IOCs. Deterministic: the same text always gives the same ordered output."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = refang(text[:MAX_TEXT_CHARS])
    found: Dict[Tuple[str, str], IOC] = {}
    excluded: Dict[Tuple[str, str], str] = {}
    consumed: List[Tuple[int, int]] = []  # spans already claimed by URLs / e-mails / hashes

    def overlaps(s: int, e: int) -> bool:
        return any(s < ce and e > cs for cs, ce in consumed)

    def add(kind: str, value: str, start: int, end: int) -> None:
        key = (kind, value)
        if key in found:
            found[key].count += 1
        else:
            found[key] = IOC(type=kind, value=value, defanged=defang(value) if kind in ("url", "domain", "ipv4", "email") else value,
                             context=_context(text, start, end))

    for m in _URL.finditer(text):
        url = m.group(0).rstrip(".,;:!?")
        host = re.sub(r"^[a-z]+://", "", url, flags=re.I).split("/")[0].split(":")[0]
        consumed.append((m.start(), m.start() + len(url)))
        if not include_reference_domains and _is_reference(host):
            excluded[("url", url)] = "well-known reference domain"
            continue
        add("url", url, m.start(), m.start() + len(url))
        if not _IPV4.fullmatch(host) and "." in host:
            add("domain", host.lower(), m.start(), m.start() + len(url))

    for m in _EMAIL.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        consumed.append((m.start(), m.end()))
        add("email", m.group(0).lower(), m.start(), m.end())

    for kind, pat in _HASHES:
        for m in pat.finditer(text):
            if overlaps(m.start(), m.end()):
                continue
            consumed.append((m.start(), m.end()))
            add(kind, m.group(0).lower(), m.start(), m.end())

    for m in _CVE.finditer(text):
        add("cve", m.group(0).upper(), m.start(), m.end())

    for m in _IPV4.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        ip = m.group(0)
        reason = _ip_exclusion(ip)
        if reason:
            excluded[("ipv4", ip)] = reason
            continue
        consumed.append((m.start(), m.end()))
        add("ipv4", ip, m.start(), m.end())

    for m in _DOMAIN.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        dom = m.group(0).lower().rstrip(".")
        tld = dom.rsplit(".", 1)[-1]
        if dom in _FALSE_DOMAINS or len(dom) < 4:
            continue
        if tld in _FILE_EXTENSIONS:
            excluded[("domain", dom)] = "file name, not a domain"
            continue
        if not include_reference_domains and _is_reference(dom):
            excluded[("domain", dom)] = "well-known reference domain"
            continue
        add("domain", dom, m.start(), m.end())

    order = {"ipv4": 0, "domain": 1, "url": 2, "email": 3, "sha256": 4, "sha1": 5, "md5": 6, "cve": 7}
    iocs = sorted(found.values(), key=lambda i: (order.get(i.type, 99), i.value))
    excl = [{"type": k, "value": v, "reason": r} for (k, v), r in sorted(excluded.items())]
    return IOCResult(iocs=iocs, excluded=excl)


# ------------------------------------------------------------------ exports
_CSV_COLUMNS = ["type", "value", "defanged", "count", "context"]


def to_csv(result: IOCResult) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_CSV_COLUMNS)
    for i in result.iocs:
        w.writerow([i.type, i.value, i.defanged, i.count, i.context])
    return buf.getvalue()


def _stix_pattern(i: IOC) -> Optional[str]:
    v = i.value.replace("\\", "\\\\").replace("'", "\\'")
    return {
        "ipv4": f"[ipv4-addr:value = '{v}']",
        "domain": f"[domain-name:value = '{v}']",
        "url": f"[url:value = '{v}']",
        "email": f"[email-addr:value = '{v}']",
        "md5": f"[file:hashes.MD5 = '{v}']",
        "sha1": f"[file:hashes.'SHA-1' = '{v}']",
        "sha256": f"[file:hashes.'SHA-256' = '{v}']",
    }.get(i.type)


def to_stix_bundle(result: IOCResult, report_title: str = "Analyst report") -> Dict:
    """STIX 2.1 bundle: one Indicator per network/file IOC, one Vulnerability per CVE.

    Ids are deterministic (uuid5 of type+value) so re-exporting the same report
    does not create duplicate objects downstream.
    """
    import json
    import uuid
    from datetime import datetime, timezone

    import stix2

    ns = uuid.UUID("2b7c6d1e-4a3f-4f8e-9d21-6c5b4a3e2f10")
    created = datetime.now(timezone.utc).replace(microsecond=0)
    objects = []
    for i in result.iocs:
        oid = uuid.uuid5(ns, f"{i.type}:{i.value}")
        if i.type == "cve":
            objects.append(stix2.Vulnerability(
                id=f"vulnerability--{oid}", name=i.value, created=created, modified=created,
                external_references=[{"source_name": "cve", "external_id": i.value}],
            ))
            continue
        pattern = _stix_pattern(i)
        if pattern is None:
            continue
        objects.append(stix2.Indicator(
            id=f"indicator--{oid}", name=f"{i.type}: {i.value}"[:250], pattern=pattern, pattern_type="stix",
            valid_from=created, created=created, modified=created,
            description=f"Extracted automatically from '{report_title[:200]}'. Review before blocking.",
            indicator_types=["unknown"],
        ))
    bundle = stix2.Bundle(objects=objects, allow_custom=False) if objects else None
    if bundle is None:
        return {"type": "bundle", "id": f"bundle--{uuid.uuid4()}", "objects": []}
    return json.loads(bundle.serialize())
