"""LLM-Powered Cybersecurity Knowledge Graph Extractor.

Extracts typed cybersecurity entities and relational assertions from unstructured
CTI text using a Large Language Model (Qwen2.5 / Llama-3 / local endpoint) with
guaranteed fallback to clause-aware rule extraction when offline.
Inserts verified extractions directly into the SQLite Knowledge Graph DB.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.llm.llm_client import LocalLLMClient
from src.graph.db import CTIKnowledgeGraphDB
from src.nlp.relation_extractor import CyberRelationExtractor
from src.preprocessor.segmenter import PreprocessedSentence

logger = logging.getLogger(__name__)

LLM_KG_SYSTEM_PROMPT = """You are an expert Cyber Threat Intelligence (CTI) Knowledge Graph Extractor.
Extract cybersecurity entities and relationships from the provided text according to this ontology:

Allowed Entity Categories:
- ThreatActor (e.g. APT28, Lazarus, FIN7)
- Malware (e.g. Zebrocy, TrickBot, Emotet)
- Tool (e.g. Cobalt Strike, Mimikatz)
- Technique (e.g. Spearphishing, DLL Sideloading)
- Organization (e.g. NATO, European Commission, banks)
- Location (e.g. Ukraine, Germany, Europe)
- CVE (e.g. CVE-2021-44228)

Allowed Relations:
- USES: ThreatActor/Malware -> Malware/Tool
- TARGETS: ThreatActor/Malware -> Organization/Location
- EXPLOITS: ThreatActor/Malware/Tool -> CVE
- USES_TECHNIQUE: ThreatActor/Malware -> Technique

Respond ONLY with a valid JSON object matching this schema:
{
  "entities": [
    {"name": "...", "category": "..."}
  ],
  "relations": [
    {
      "subject": "...",
      "subject_category": "...",
      "predicate": "USES | TARGETS | EXPLOITS | USES_TECHNIQUE",
      "object": "...",
      "object_category": "...",
      "trigger": "verb or phrase establishing link",
      "evidence_sentence": "verbatim text segment",
      "confidence": 0.85
    }
  ]
}
Do not hallucinate. If no clear relations exist, return empty lists.
"""


class LLMKnowledgeGraphExtractor:
    """Extracts entities and relationships using LLM inference with database commitment."""

    def __init__(
        self,
        llm_client: Optional[LocalLLMClient] = None,
        db: Optional[CTIKnowledgeGraphDB] = None,
        fallback_extractor: Optional[CyberRelationExtractor] = None,
    ):
        self.llm = llm_client or LocalLLMClient()
        self.db = db or CTIKnowledgeGraphDB()
        self.fallback = fallback_extractor or CyberRelationExtractor()

    def extract_from_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        commit_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Extract entities and relationships from text using LLM, with graceful offline fallback."""
        meta = metadata or {}
        report_id = meta.get("report_id", "EXTRACTED-REPORT")
        source_vendor = meta.get("source", "CTI-Analyst")
        pub_date = meta.get("publication_date", "2024-01-01")
        sentence_id = meta.get("sentence_id", "s001")

        llm_available = False
        try:
            llm_available = self.llm.is_available()
        except Exception:
            llm_available = False

        if llm_available:
            try:
                raw_resp = self.llm.chat_complete(
                    system_prompt=LLM_KG_SYSTEM_PROMPT,
                    user_prompt=f"Extract CTI knowledge graph elements from this text:\n\n\"{text}\"",
                    temperature=0.1,
                    max_tokens=600,
                )
                if raw_resp:
                    clean_json = self._extract_json_block(raw_resp)
                    data = json.loads(clean_json)
                    return self._process_and_save(data, text, meta, commit_to_db, method="llm")
            except Exception as exc:
                logger.warning("LLM extraction failed (%s); falling back to rule extractor", exc)

        # Fallback to deterministic CTI relation extractor
        return self._extract_with_fallback(text, meta, commit_to_db)

    def _extract_json_block(self, text: str) -> str:
        """Extract JSON substring from markdown fences or raw string."""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]
        return text

    def _process_and_save(
        self,
        data: Dict[str, Any],
        raw_text: str,
        meta: Dict[str, Any],
        commit_to_db: bool,
        method: str = "llm",
    ) -> Dict[str, Any]:
        """Normalize extracted entities/relations and optionally commit to SQLite."""
        entities = data.get("entities", [])
        relations = data.get("relations", [])

        report_id = meta.get("report_id", "EXTRACTED-REPORT")
        source = meta.get("source", "CTI-Analyst")
        date = meta.get("publication_date", "2024-01-01")
        s_id = meta.get("sentence_id", "s001")

        saved_entities = []
        for e in entities:
            name = str(e.get("name", "")).strip()
            category = str(e.get("category", "Entity")).strip()
            if not name:
                continue
            e_id = f"{category}::{name}"
            saved_entities.append({"id": e_id, "name": name, "category": category})
            if commit_to_db:
                self.db.insert_entity(
                    entity_id=e_id,
                    name=name,
                    category=category,
                    mention_count=1,
                    first_seen=date,
                    last_seen=date,
                    sources=[source],
                    reports=[report_id],
                )

        saved_relations = []
        for idx, r in enumerate(relations, start=1):
            s_name = str(r.get("subject", "")).strip()
            s_cat = str(r.get("subject_category", "Entity")).strip()
            s_id_str = f"{s_cat}::{s_name}"

            o_name = str(r.get("object", "")).strip()
            o_cat = str(r.get("object_category", "Entity")).strip()
            o_id_str = f"{o_cat}::{o_name}"

            pred = str(r.get("predicate", "RELATED_TO")).strip().upper()
            ev = str(r.get("evidence_sentence") or raw_text).strip()
            conf = float(r.get("confidence", 0.85))
            trigger = str(r.get("trigger", "llm_inferred")).strip()
            rel_id = f"LLM-{report_id}-{idx:04d}"

            rel_record = {
                "id": rel_id,
                "subject_id": s_id_str,
                "subject_name": s_name,
                "predicate": pred,
                "object_id": o_id_str,
                "object_name": o_name,
                "evidence_text": ev,
                "confidence": conf,
                "extraction_method": method,
            }
            saved_relations.append(rel_record)

            if commit_to_db:
                self.db.insert_relationship(
                    rel_id=rel_id,
                    subject_id=s_id_str,
                    subject_name=s_name,
                    subject_category=s_cat,
                    predicate=pred,
                    object_id=o_id_str,
                    object_name=o_name,
                    object_category=o_cat,
                    evidence_text=ev,
                    confidence=conf,
                    trigger=trigger,
                    sentence_id=s_id,
                    report_id=report_id,
                    source_vendor=source,
                    publication_date=date,
                    extraction_method=method,
                )

        return {
            "method": method,
            "entities": saved_entities,
            "relationships": saved_relations,
            "total_entities": len(saved_entities),
            "total_relationships": len(saved_relations),
        }

    def _extract_with_fallback(
        self,
        text: str,
        meta: Dict[str, Any],
        commit_to_db: bool,
    ) -> Dict[str, Any]:
        """Run deterministic clause-aware extractor and format output."""
        pseudo_sent = PreprocessedSentence(
            sentence_id=meta.get("sentence_id", "s001"),
            paragraph_id=meta.get("paragraph_id", "p001"),
            report_id=meta.get("report_id", "EXTRACTED-REPORT"),
            source=meta.get("source", "CTI-Analyst"),
            publication_date=meta.get("publication_date", "2024-01-01"),
            paragraph_index=0,
            sentence_index=0,
            global_sentence_index=0,
            text=text,
            start_char=0,
            end_char=len(text),
        )
        ents = self.fallback.ner_engine.extract_from_sentence(pseudo_sent)
        rels = self.fallback.extract_from_sentence(sentence=pseudo_sent, entities=ents)

        data = {
            "entities": [{"name": e.canonical_name, "category": e.category} for e in ents],
            "relations": [
                {
                    "subject": r.source_entity,
                    "subject_category": r.source_category,
                    "predicate": r.relation_type,
                    "object": r.target_entity,
                    "object_category": r.target_category,
                    "trigger": r.trigger,
                    "evidence_sentence": r.evidence_text,
                    "confidence": r.confidence,
                }
                for r in rels
            ],
        }
        return self._process_and_save(data, text, meta, commit_to_db, method="rule_based_fallback")
