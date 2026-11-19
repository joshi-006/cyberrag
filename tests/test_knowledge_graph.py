"""Tests for Knowledge Graph construction, export, and analytics."""

import json
from pathlib import Path
import networkx as nx
import pytest

from src.graph.builder import CyberKnowledgeGraphBuilder
from src.graph.exporter import CyberKnowledgeGraphExporter
from src.graph.analytics import CyberKnowledgeGraphAnalytics
from src.nlp.ner_engine import ExtractedEntity
from src.nlp.relation_extractor import ExtractedRelation


@pytest.fixture
def sample_graph_data():
    entities = [
        ExtractedEntity(
            entity_id="ThreatActor::Volt Typhoon",
            text="Volt Typhoon",
            category="ThreatActor",
            canonical_name="Volt Typhoon",
            canonical_id="TA-CN-0001",
            start_char=0,
            end_char=12,
            confidence=0.98,
            sentence_id="R1::p0::s0",
            paragraph_id="R1::p0",
            report_id="R1",
            source="CISA",
            publication_date="2023-12-13",
        ),
        ExtractedEntity(
            entity_id="CVE::CVE-2024-21887",
            text="CVE-2024-21887",
            category="CVE",
            canonical_name="CVE-2024-21887",
            canonical_id="CVE-2024-21887",
            start_char=23,
            end_char=37,
            confidence=0.99,
            sentence_id="R1::p0::s0",
            paragraph_id="R1::p0",
            report_id="R1",
            source="CISA",
            publication_date="2023-12-13",
        ),
    ]

    relations = [
        ExtractedRelation(
            relation_id="rel::Volt Typhoon::EXPLOITS::CVE-2024-21887::R1::p0::s0",
            relation_type="EXPLOITS",
            source_entity="Volt Typhoon",
            source_category="ThreatActor",
            source_id="TA-CN-0001",
            target_entity="CVE-2024-21887",
            target_category="CVE",
            target_id="CVE-2024-21887",
            confidence=0.95,
            trigger="exploited",
            evidence_text="Volt Typhoon exploited CVE-2024-21887.",
            sentence_id="R1::p0::s0",
            paragraph_id="R1::p0",
            report_id="R1",
            source="CISA",
            publication_date="2023-12-13",
        )
    ]
    return entities, relations


def test_kg_builder_and_analytics(sample_graph_data, tmp_path):
    entities, relations = sample_graph_data
    builder = CyberKnowledgeGraphBuilder()
    kg = builder.build(entities, relations)

    assert kg.number_of_nodes() == 2
    assert kg.number_of_edges() == 1

    # Edge check
    u = "ThreatActor::Volt Typhoon"
    v = "CVE::CVE-2024-21887"
    assert kg.has_edge(u, v, key="EXPLOITS")
    edge = kg[u][v]["EXPLOITS"]
    assert edge["evidence_count"] == 1
    assert edge["confidence"] == 0.95

    # Analytics test
    analytics = CyberKnowledgeGraphAnalytics(kg)
    summary = analytics.compute_summary()
    assert summary["graph_metrics"]["total_nodes"] == 2
    assert summary["graph_metrics"]["total_edges"] == 1
    assert len(summary["threat_actors_profiled"]) == 1

    # Exporters test
    exporter = CyberKnowledgeGraphExporter(kg)
    json_path = tmp_path / "test_kg.json"
    graphml_path = tmp_path / "test_kg.graphml"
    cypher_path = tmp_path / "test_kg.cypher"

    exporter.export_json(str(json_path))
    exporter.export_graphml(str(graphml_path))
    exporter.export_cypher(str(cypher_path))

    assert json_path.exists()
    assert graphml_path.exists()
    assert cypher_path.exists()

    with open(cypher_path, "r", encoding="utf-8") as f:
        cypher_text = f.read()
    assert "MERGE (n:ThreatActor {name: 'Volt Typhoon'})" in cypher_text
    assert "MERGE (a)-[r:EXPLOITS]->(b)" in cypher_text
