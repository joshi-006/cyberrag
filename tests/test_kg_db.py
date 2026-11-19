"""Unit tests for SQLite CTI Knowledge Graph Database and LLM Extractor."""
import tempfile
import unittest
from pathlib import Path

from src.graph.db import CTIKnowledgeGraphDB
from src.graph.llm_extractor import LLMKnowledgeGraphExtractor


class TestCTIKnowledgeGraphDB(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_kg.db"
        self.db = CTIKnowledgeGraphDB(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_schema_initialization(self):
        """Verify entities and relationships tables exist."""
        stats = self.db.get_stats()
        self.assertEqual(stats["total_entities"], 0)
        self.assertEqual(stats["total_relationships"], 0)

    def test_02_insert_and_get_entity(self):
        """Verify entity insertion, deduplication and retrieval."""
        self.db.insert_entity(
            entity_id="ThreatActor::APT28",
            name="APT28",
            category="ThreatActor",
            canonical_id="G0007",
            mention_count=2,
            first_seen="2018-01-01",
            last_seen="2021-05-01",
            sources=["Mandiant", "CrowdStrike"],
            surface_forms=["APT28", "Fancy Bear"],
        )
        ent = self.db.get_entity("APT28")
        self.assertIsNotNone(ent)
        self.assertEqual(ent["id"], "ThreatActor::APT28")
        self.assertEqual(ent["category"], "ThreatActor")
        self.assertEqual(ent["canonical_id"], "G0007")
        self.assertEqual(ent["mention_count"], 2)
        self.assertIn("Mandiant", ent["sources"])
        self.assertIn("Fancy Bear", ent["surface_forms"])

    def test_03_insert_and_query_relationship(self):
        """Verify relationship insertion and filtered queries."""
        self.db.insert_entity(entity_id="ThreatActor::APT28", name="APT28", category="ThreatActor")
        self.db.insert_entity(entity_id="Malware::Zebrocy", name="Zebrocy", category="Malware")

        self.db.insert_relationship(
            rel_id="R0001",
            subject_id="ThreatActor::APT28",
            subject_name="APT28",
            subject_category="ThreatActor",
            predicate="USES",
            object_id="Malware::Zebrocy",
            object_name="Zebrocy",
            object_category="Malware",
            evidence_text="APT28 used Zebrocy malware in recent attacks.",
            confidence=0.92,
            trigger="used",
            source_vendor="ESET",
            publication_date="2020-04-15",
        )

        # Query by subject
        rels = self.db.find_relationships(subject="APT28")
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["predicate"], "USES")
        self.assertEqual(rels[0]["object_name"], "Zebrocy")

        # Query by predicate
        rels_uses = self.db.find_relationships(predicate="USES")
        self.assertEqual(len(rels_uses), 1)

        # Query non-existent
        rels_targets = self.db.find_relationships(predicate="TARGETS")
        self.assertEqual(len(rels_targets), 0)

    def test_04_neighborhood_extraction(self):
        """Verify subgraph extraction for UI rendering."""
        self.db.insert_entity(entity_id="ThreatActor::APT28", name="APT28", category="ThreatActor")
        self.db.insert_entity(entity_id="Malware::Zebrocy", name="Zebrocy", category="Malware")
        self.db.insert_entity(entity_id="Organization::NATO", name="NATO", category="Organization")

        self.db.insert_relationship(
            rel_id="R1",
            subject_id="ThreatActor::APT28",
            subject_name="APT28",
            subject_category="ThreatActor",
            predicate="USES",
            object_id="Malware::Zebrocy",
            object_name="Zebrocy",
            object_category="Malware",
            evidence_text="APT28 used Zebrocy.",
        )
        self.db.insert_relationship(
            rel_id="R2",
            subject_id="ThreatActor::APT28",
            subject_name="APT28",
            subject_category="ThreatActor",
            predicate="TARGETS",
            object_id="Organization::NATO",
            object_name="NATO",
            object_category="Organization",
            evidence_text="APT28 targeted NATO.",
        )

        subgraph = self.db.get_neighborhood("ThreatActor::APT28", max_hops=1)
        self.assertEqual(len(subgraph["nodes"]), 3)
        self.assertEqual(len(subgraph["edges"]), 2)

    def test_05_export_to_networkx(self):
        """Verify database exports seamlessly back to NetworkX MultiDiGraph."""
        self.db.insert_entity(entity_id="ThreatActor::APT28", name="APT28", category="ThreatActor")
        self.db.insert_entity(entity_id="Malware::Zebrocy", name="Zebrocy", category="Malware")
        self.db.insert_relationship(
            rel_id="R1",
            subject_id="ThreatActor::APT28",
            subject_name="APT28",
            subject_category="ThreatActor",
            predicate="USES",
            object_id="Malware::Zebrocy",
            object_name="Zebrocy",
            object_category="Malware",
            evidence_text="APT28 used Zebrocy.",
        )

        G = self.db.export_to_networkx()
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertEqual(G.number_of_edges(), 1)
        self.assertTrue(G.has_edge("ThreatActor::APT28", "Malware::Zebrocy", key="USES"))

    def test_06_llm_extractor_fallback(self):
        """Verify LLM extractor gracefully falls back to rule extraction and stores in DB."""
        extractor = LLMKnowledgeGraphExtractor(db=self.db)
        text = "APT28 deployed Zebrocy malware across European government networks."
        result = extractor.extract_from_text(
            text=text,
            metadata={"source": "ESET", "publication_date": "2020-04-15"},
            commit_to_db=True,
        )

        self.assertGreaterEqual(result["total_entities"], 1)
        # Check DB was populated
        stats = self.db.get_stats()
        self.assertGreaterEqual(stats["total_entities"], 1)


if __name__ == "__main__":
    unittest.main()
