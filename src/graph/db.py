"""SQLite-backed Persistent Knowledge Graph Database for CTI.

Replaces static NetworkX JSON storage with a relational, ACID-compliant SQLite
database schema while maintaining full backward-compatibility for graph traversal
and frontend visualization.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "cti_kg.db"


class CTIKnowledgeGraphDB:
    """Persistent SQLite database manager for CTI entities, relationships, and assertions."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_schema(self) -> None:
        """Create entities and relationships tables with optimized indices."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    canonical_id TEXT DEFAULT '',
                    mention_count INTEGER DEFAULT 1,
                    first_seen TEXT DEFAULT '',
                    last_seen TEXT DEFAULT '',
                    sources TEXT DEFAULT '[]',
                    reports TEXT DEFAULT '[]',
                    surface_forms TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    subject_name TEXT NOT NULL,
                    subject_category TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_name TEXT NOT NULL,
                    object_category TEXT NOT NULL,
                    confidence REAL DEFAULT 0.85,
                    confidence_level TEXT DEFAULT 'HIGH',
                    polarity TEXT DEFAULT 'positive',
                    trigger TEXT DEFAULT '',
                    evidence_text TEXT NOT NULL,
                    sentence_id TEXT DEFAULT '',
                    paragraph_id TEXT DEFAULT '',
                    report_id TEXT DEFAULT '',
                    source_vendor TEXT DEFAULT '',
                    publication_date TEXT DEFAULT '',
                    extraction_method TEXT DEFAULT 'llm',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
                CREATE INDEX IF NOT EXISTS idx_entities_category ON entities(category);
                CREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(subject_id);
                CREATE INDEX IF NOT EXISTS idx_rel_object ON relationships(object_id);
                CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relationships(predicate);
                CREATE INDEX IF NOT EXISTS idx_rel_sub_pred ON relationships(subject_id, predicate);
                CREATE INDEX IF NOT EXISTS idx_rel_source_date ON relationships(source_vendor, publication_date);
            """)

    def insert_entity(
        self,
        entity_id: str,
        name: str,
        category: str,
        canonical_id: str = "",
        mention_count: int = 1,
        first_seen: str = "",
        last_seen: str = "",
        sources: Optional[List[str]] = None,
        reports: Optional[List[str]] = None,
        surface_forms: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update an entity record."""
        sources_json = json.dumps(sources or [])
        reports_json = json.dumps(reports or [])
        surfaces_json = json.dumps(surface_forms or [name])
        meta_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO entities (
                    id, name, category, canonical_id, mention_count,
                    first_seen, last_seen, sources, reports, surface_forms, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    mention_count = entities.mention_count + excluded.mention_count,
                    first_seen = CASE
                        WHEN entities.first_seen = '' OR excluded.first_seen < entities.first_seen
                        THEN excluded.first_seen ELSE entities.first_seen END,
                    last_seen = CASE
                        WHEN entities.last_seen = '' OR excluded.last_seen > entities.last_seen
                        THEN excluded.last_seen ELSE entities.last_seen END,
                    canonical_id = CASE
                        WHEN excluded.canonical_id != '' THEN excluded.canonical_id ELSE entities.canonical_id END
            """, (
                entity_id, name, category, canonical_id, mention_count,
                first_seen, last_seen, sources_json, reports_json, surfaces_json, meta_json
            ))

    def insert_relationship(
        self,
        rel_id: str,
        subject_id: str,
        subject_name: str,
        subject_category: str,
        predicate: str,
        object_id: str,
        object_name: str,
        object_category: str,
        evidence_text: str,
        confidence: float = 0.85,
        polarity: str = "positive",
        trigger: str = "",
        sentence_id: str = "",
        paragraph_id: str = "",
        report_id: str = "",
        source_vendor: str = "",
        publication_date: str = "",
        extraction_method: str = "llm",
    ) -> None:
        """Insert a verified semantic relationship / assertion."""
        conf_level = "HIGH" if confidence >= 0.85 else ("MEDIUM" if confidence >= 0.60 else "LOW")

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO relationships (
                    id, subject_id, subject_name, subject_category, predicate,
                    object_id, object_name, object_category, confidence, confidence_level,
                    polarity, trigger, evidence_text, sentence_id, paragraph_id,
                    report_id, source_vendor, publication_date, extraction_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rel_id, subject_id, subject_name, subject_category, predicate,
                object_id, object_name, object_category, confidence, conf_level,
                polarity, trigger, evidence_text, sentence_id, paragraph_id,
                report_id, source_vendor, publication_date, extraction_method
            ))

    def get_entity(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """Look up entity by canonical ID or name (case-insensitive)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM entities WHERE id = ? OR name = ? COLLATE NOCASE LIMIT 1",
                (name_or_id, name_or_id)
            )
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d["sources"] = json.loads(d.get("sources") or "[]")
            d["reports"] = json.loads(d.get("reports") or "[]")
            d["surface_forms"] = json.loads(d.get("surface_forms") or "[]")
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            return d

    def find_relationships(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        polarity: str = "positive",
    ) -> List[Dict[str, Any]]:
        """Query relationships with optional subject, predicate, and object filters."""
        clauses = ["polarity = ?"]
        params: List[Any] = [polarity]

        if subject:
            clauses.append("(subject_id = ? OR subject_name = ? COLLATE NOCASE)")
            params.extend([subject, subject])
        if predicate:
            clauses.append("predicate = ? COLLATE NOCASE")
            params.append(predicate)
        if obj:
            clauses.append("(object_id = ? OR object_name = ? COLLATE NOCASE)")
            params.extend([obj, obj])

        sql = f"SELECT * FROM relationships WHERE {' AND '.join(clauses)} ORDER BY confidence DESC"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def get_neighborhood(self, entity_id: str, max_hops: int = 1) -> Dict[str, Any]:
        """Extract a 1-hop or 2-hop neighborhood subgraph for graph visualization."""
        visited_nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        root = self.get_entity(entity_id)
        if not root:
            return {"nodes": [], "edges": []}
        visited_nodes[root["id"]] = root

        current_frontier = {root["id"]}
        for _ in range(max_hops):
            if not current_frontier:
                break
            next_frontier = set()
            with self._get_connection() as conn:
                placeholders = ",".join("?" for _ in current_frontier)
                sql = f"""
                    SELECT * FROM relationships
                    WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})
                """
                cursor = conn.execute(sql, tuple(current_frontier) * 2)
                for row in cursor.fetchall():
                    rel = dict(row)
                    edges.append({
                        "from": rel["subject_id"],
                        "to": rel["object_id"],
                        "label": rel["predicate"],
                        "relation_type": rel["predicate"],
                        "confidence": rel["confidence"],
                        "evidence_text": rel["evidence_text"],
                        "source": rel["source_vendor"],
                        "date": rel["publication_date"],
                    })
                    for nid in (rel["subject_id"], rel["object_id"]):
                        if nid not in visited_nodes:
                            node_data = self.get_entity(nid)
                            if node_data:
                                visited_nodes[nid] = node_data
                            next_frontier.add(nid)
            current_frontier = next_frontier

        return {
            "nodes": [
                {
                    "id": n["id"],
                    "label": n["name"],
                    "category": n["category"],
                    "type": n["category"],
                    "canonical_id": n.get("canonical_id", ""),
                    "mention_count": n.get("mention_count", 1),
                }
                for n in visited_nodes.values()
            ],
            "edges": edges,
        }

    def export_to_networkx(self) -> nx.MultiDiGraph:
        """Export SQL database contents to a NetworkX MultiDiGraph for graph algorithms."""
        G = nx.MultiDiGraph()
        with self._get_connection() as conn:
            for row in conn.execute("SELECT * FROM entities"):
                G.add_node(
                    row["id"],
                    name=row["name"],
                    category=row["category"],
                    canonical_id=row["canonical_id"],
                    mention_count=row["mention_count"],
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                )
            for row in conn.execute("SELECT * FROM relationships"):
                G.add_edge(
                    row["subject_id"],
                    row["object_id"],
                    key=row["predicate"],
                    relation_type=row["predicate"],
                    confidence=row["confidence"],
                    evidence_text=row["evidence_text"],
                    source=row["source_vendor"],
                    publication_date=row["publication_date"],
                    report_id=row["report_id"],
                )
        return G

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of nodes and relationships."""
        with self._get_connection() as conn:
            total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            total_relations = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            categories = dict(conn.execute(
                "SELECT category, COUNT(*) FROM entities GROUP BY category"
            ).fetchall())
            predicates = dict(conn.execute(
                "SELECT predicate, COUNT(*) FROM relationships GROUP BY predicate"
            ).fetchall())

        return {
            "database_path": str(self.db_path),
            "total_entities": total_entities,
            "total_relationships": total_relations,
            "category_distribution": categories,
            "predicate_distribution": predicates,
        }
