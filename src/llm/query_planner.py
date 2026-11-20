"""
LLM Query Planner for CyberRAG.
Decomposes unstructured analyst queries into structured query plans (sub-claims, canonical predicates, temporal scope).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.llm.llm_client import LocalLLMClient
from src.llm.query_expander import QueryExpander, KNOWN_ALIASES


class QueryIntent(str, Enum):
    FACTUAL_VERIFICATION = "FACTUAL_VERIFICATION"
    RELATIONAL_QUERY = "RELATIONAL_QUERY"
    TEMPORAL_EVOLUTION = "TEMPORAL_EVOLUTION"
    GENERAL_EXPLORATION = "GENERAL_EXPLORATION"


# Canonical KG Schema Predicates
CANONICAL_PREDICATES = {"USES", "TARGETS", "EXPLOITS", "USES_TECHNIQUE"}

# Mapping common natural-language relation phrases to canonical predicates
RELATION_MAP = {
    "use": "USES",
    "uses": "USES",
    "used": "USES",
    "using": "USES",
    "operate": "USES",
    "operates": "USES",
    "operated": "USES",
    "operating": "USES",
    "deploy": "USES",
    "deploys": "USES",
    "deployed": "USES",
    "deploying": "USES",
    "deliver": "USES",
    "delivers": "USES",
    "delivered": "USES",
    "delivering": "USES",
    "employ": "USES",
    "employs": "USES",
    "employed": "USES",
    "associated with": "USES",
    "associated": "USES",
    "linked to": "USES",
    "target": "TARGETS",
    "targets": "TARGETS",
    "targeted": "TARGETS",
    "targeting": "TARGETS",
    "exploit": "EXPLOITS",
    "exploits": "EXPLOITS",
    "exploited": "EXPLOITS",
    "exploiting": "EXPLOITS",
    "use technique": "USES_TECHNIQUE",
    "uses technique": "USES_TECHNIQUE",
    "leverage": "USES_TECHNIQUE",
    "leverages": "USES_TECHNIQUE",
    "leveraged": "USES_TECHNIQUE",
    "leverage technique": "USES_TECHNIQUE",
    "leverages technique": "USES_TECHNIQUE",
}


@dataclass
class SubClaim:
    subject: str
    predicate: str  # Must be one of CANONICAL_PREDICATES
    object: str
    raw_relation: str = ""
    polarity: str = "positive"

    def to_natural_phrase(self) -> str:
        if self.polarity == "negative":
            pred_map = {
                "USES": "does not use",
                "TARGETS": "does not target",
                "EXPLOITS": "does not exploit",
                "USES_TECHNIQUE": "does not use technique",
            }
            verb = pred_map.get(self.predicate, f"does not {self.predicate.lower()}")
            return f"{self.subject} {verb} {self.object}"
        pred_map = {
            "USES": "uses",
            "TARGETS": "targets",
            "EXPLOITS": "exploits",
            "USES_TECHNIQUE": "uses technique",
        }
        verb = pred_map.get(self.predicate, self.predicate.lower())
        return f"{self.subject} {verb} {self.object}"


@dataclass
class QueryPlan:
    raw_query: str
    intent: QueryIntent
    entities: List[str]
    aliases: Dict[str, List[str]]
    sub_claims: List[SubClaim]
    temporal_scope: Optional[Tuple[str, str]] = None  # (start_date, end_date) in YYYY-MM-DD
    search_queries: List[str] = field(default_factory=list)


class LLMQueryPlanner:
    """Decomposes queries into entities, canonical relations, temporal scopes, and sub-claims."""

    def __init__(
        self,
        expander: Optional[QueryExpander] = None,
        llm_client: Optional[LocalLLMClient] = None,
    ):
        self.expander = expander or QueryExpander()
        self.llm_client = llm_client or LocalLLMClient()

    def plan(self, query: str) -> QueryPlan:
        if self.llm_client.is_available():
            llm_plan = self._plan_with_llm(query)
            if llm_plan:
                return llm_plan

        return self._plan_heuristic(query)

    def _plan_with_llm(self, query: str) -> Optional[QueryPlan]:
        system = (
            "You are a CTI query decomposition planner. Extract structured JSON only:\n"
            "{\n"
            '  "intent": "FACTUAL_VERIFICATION" | "RELATIONAL_QUERY" | "TEMPORAL_EVOLUTION" | "GENERAL_EXPLORATION",\n'
            '  "subject": string,\n'
            '  "predicate": "USES" | "TARGETS" | "EXPLOITS" | "USES_TECHNIQUE",\n'
            '  "object": string,\n'
            '  "temporal_scope": [start_year_str, end_year_str] or null\n'
            "}\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. 'subject' and 'object' MUST BE EXACT verbatim substrings copied directly from the query. NEVER rename, spell-correct, normalize, or alter entity names.\n"
            "2. 'predicate' rules:\n"
            "   - 'TARGETS': when an actor attacks, campaigns against, or targets an organization, sector, or country (e.g. 'FIN7 target hospitality' -> 'TARGETS').\n"
            "   - 'USES': when an actor uses, deploys, or operates malware or tools (e.g. 'APT28 use Zebrocy' -> 'USES').\n"
            "   - 'EXPLOITS': when an actor exploits a CVE or vulnerability.\n"
            "   - 'USES_TECHNIQUE': when an actor leverages an ATT&CK technique.\n"
        )
        resp = self.llm_client.chat_complete(system, f"Query: {query}")
        if not resp:
            return None
        try:
            match = re.search(r"\{.*\}", resp, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))
            intent = QueryIntent(data.get("intent", "FACTUAL_VERIFICATION"))
            sub = data.get("subject", "").strip()
            pred = data.get("predicate", "USES")
            obj = data.get("object", "").strip()
            if not sub or not obj:
                return None

            # Guardrail 0: Enforce intent consistency with explicit grammatical triggers
            if re.search(r"^(does|did|is|was|whether|can|has|have)\b", query.strip(), re.I):
                intent = QueryIntent.FACTUAL_VERIFICATION
            elif re.search(r"\b(evolve|history|timeline|between\s+\d{4}|before\s+\d{4}|after\s+\d{4})\b", query, re.I):
                intent = QueryIntent.TEMPORAL_EVOLUTION
            elif re.search(r"^(what|which|who|list)\b", query.strip(), re.I):
                intent = QueryIntent.RELATIONAL_QUERY

            # Guardrail 1: Enforce exact substring presence in query to eliminate hallucinations
            s_found = sub.lower() in query.lower()
            o_found = obj.lower() in query.lower()
            if not s_found or not o_found:
                # LLM rewrote or invented an entity (e.g. Zebrocy -> Zerobility)
                return self._plan_heuristic(query)

            # Preserve exact query casing
            s_m = re.search(rf"\b{re.escape(sub)}\b", query, re.I)
            if s_m:
                sub = s_m.group(0)
            o_m = re.search(rf"\b{re.escape(obj)}\b", query, re.I)
            if o_m:
                obj = o_m.group(0)

            # Guardrail 2: Enforce predicate consistency with explicit trigger keywords
            if re.search(r"\b(target|targets|targeted|targeting|against)\b", query, re.I):
                pred = "TARGETS"
            elif re.search(r"\b(exploit|exploits|exploited|exploiting)\b", query, re.I):
                pred = "EXPLOITS"
            elif re.search(r"\b(use\s+technique|uses\s+technique|leverage\s+technique|leverages\s+technique)\b", query, re.I):
                pred = "USES_TECHNIQUE"
            elif re.search(r"\b(use|uses|used|using|deploy|deploys|deployed|operates|operating|associated\s+with|associated|linked\s+to|connected\s+to)\b", query, re.I):
                pred = "USES"

            neg_match = re.search(r"\b(not|no\s+longer|never|doesn'?t|didn'?t|isn'?t|ceas(?:e|ed)|stop(?:ped)?)\b", query, re.I)
            polarity = "negative" if (neg_match or str(data.get("polarity", "")).lower() == "negative") else "positive"
            sub = re.sub(r"\b(not|no\s+longer|never|doesn'?t|didn'?t|isn'?t)\b", "", sub, flags=re.I).strip()
            obj = re.sub(r"\b(not|no\s+longer|never|doesn'?t|didn'?t|isn'?t)\b", "", obj, flags=re.I).strip()

            t_scope = None
            if data.get("temporal_scope") and len(data["temporal_scope"]) == 2:
                t_scope = (f"{data['temporal_scope'][0]}-01-01", f"{data['temporal_scope'][1]}-12-31")

            entities = [sub, obj]
            aliases = {e: self.expander.get_aliases(e) for e in entities if self.expander.get_aliases(e)}

            sub_claims = [SubClaim(subject=sub, predicate=pred, object=obj, polarity=polarity)]
            search_queries = [query, f"{sub} {obj}", f"{sub} {pred.lower()} {obj}"]
            return QueryPlan(
                raw_query=query,
                intent=intent,
                entities=entities,
                aliases=aliases,
                sub_claims=sub_claims,
                temporal_scope=t_scope,
                search_queries=list(dict.fromkeys(search_queries)),
            )
        except Exception:
            return None

    def _plan_heuristic(self, query: str) -> QueryPlan:
        q_lower = query.lower()

        # 1. Detect Intent
        if any(w in q_lower for w in ["history", "timeline", "before", "after", "evolve", "changed", "when"]):
            intent = QueryIntent.TEMPORAL_EVOLUTION
        elif any(w in q_lower for w in ["does", "did", "is", "whether", "verify", "true", "check", "was"]):
            intent = QueryIntent.FACTUAL_VERIFICATION
        elif any(w in q_lower for w in ["what", "which", "list", "who", "whom"]):
            intent = QueryIntent.RELATIONAL_QUERY
        else:
            intent = QueryIntent.GENERAL_EXPLORATION

        # 2. Extract Temporal Scope
        temporal_scope = self._extract_temporal_scope(query)

        # 3. Extract Entities & Canonical Relations
        entities, sub_claims = self._extract_entities_and_claims(query)

        # 4. Extract Aliases
        aliases = {e: self.expander.get_aliases(e) for e in entities if self.expander.get_aliases(e)}

        # 5. Formulate initial search queries
        search_queries = [query]
        for claim in sub_claims:
            search_queries.append(claim.to_natural_phrase())
            search_queries.append(f"{claim.subject} {claim.object}")
            if claim.polarity == "negative":
                search_queries.append(f"{claim.subject} {claim.predicate.lower()} {claim.object}")

        return QueryPlan(
            raw_query=query,
            intent=intent,
            entities=entities,
            aliases=aliases,
            sub_claims=sub_claims,
            temporal_scope=temporal_scope,
            search_queries=list(dict.fromkeys(search_queries)),
        )

    def _extract_temporal_scope(self, query: str) -> Optional[Tuple[str, str]]:
        years = [int(y) for y in re.findall(r"\b(20[12]\d)\b", query)]
        if not years:
            return None
        if len(years) == 1:
            y = years[0]
            if "before" in query.lower():
                return ("2016-01-01", f"{y}-01-01")
            elif "after" in query.lower():
                return (f"{y}-12-31", "2023-01-01")
            else:
                return (f"{y}-01-01", f"{y}-12-31")
        else:
            y_min, y_max = min(years), max(years)
            return (f"{y_min}-01-01", f"{y_max}-12-31")

    def _extract_entities_and_claims(self, query: str) -> Tuple[List[str], List[SubClaim]]:
        entities = []
        sub_claims = []

        found_rel = None
        canonical = "USES"
        for phrase, canon in sorted(RELATION_MAP.items(), key=lambda x: -len(x[0])):
            pattern = rf"\b{re.escape(phrase)}\b"
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                found_rel = phrase
                canonical = canon
                s_part = query[:m.start()].strip()
                o_part = query[m.end():].strip()

                s_clean = re.sub(r"^(does|did|is|was|whether|can)\s+", "", s_part, flags=re.IGNORECASE).strip()
                neg_match = re.search(r"\b(not|no\s+longer|never|doesn'?t|didn'?t|isn'?t|stop(?:ped)?)\b", s_clean, re.I)
                polarity = "negative" if neg_match else "positive"
                if neg_match:
                    s_clean = re.sub(r"\b(not|no\s+longer|never|doesn'?t|didn'?t|isn'?t|stop(?:ped)?)\b", "", s_clean, flags=re.IGNORECASE).strip()
                s_clean = re.sub(r"[^\w\s-]", "", s_clean).strip()
                o_clean = re.sub(r"[^\w\s-]", "", o_part).strip()

                if s_clean and o_clean:
                    entities = [s_clean, o_clean]
                    sub_claims.append(SubClaim(
                        subject=s_clean,
                        predicate=canonical,
                        object=o_clean,
                        raw_relation=found_rel,
                        polarity=polarity,
                    ))
                break

        if not entities:
            caps = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", query)
            clean_caps = [c for c in caps if c.lower() not in {"does", "did", "is", "was", "whether", "what", "which", "how"}]
            if len(clean_caps) >= 2:
                entities = clean_caps[:2]
                sub_claims.append(SubClaim(
                    subject=entities[0],
                    predicate="USES",
                    object=entities[1],
                    raw_relation="associated",
                ))
            elif clean_caps:
                entities = clean_caps

        return entities, sub_claims
