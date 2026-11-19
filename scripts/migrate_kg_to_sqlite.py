"""Migration script: Exports CTI Knowledge Graph and assertions to persistent SQLite database."""
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.graph.db import CTIKnowledgeGraphDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_kg")

GRAPH_FILE = REPO_ROOT / "data" / "cti_knowledge_graph_trusted.json"
ASSERTIONS_FILE = REPO_ROOT / "data" / "assertions.jsonl"
DB_FILE = REPO_ROOT / "data" / "cti_kg.db"


def migrate():
    logger.info("Initializing SQLite CTI Knowledge Graph at %s", DB_FILE)
    db = CTIKnowledgeGraphDB(db_path=DB_FILE)

    # 1. Migrate Entities from Graph JSON
    logger.info("Loading entities from %s...", GRAPH_FILE)
    with GRAPH_FILE.open("r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    logger.info("Migrating %d entity nodes into SQLite...", len(nodes))
    for n in nodes:
        node_id = n.get("id") or f"{n.get('category')}::{n.get('name')}"
        name = n.get("name") or n.get("label") or node_id
        category = n.get("category") or "Entity"
        db.insert_entity(
            entity_id=node_id,
            name=name,
            category=category,
            canonical_id=n.get("canonical_id", ""),
            mention_count=n.get("mention_count", 1),
            first_seen=n.get("first_seen", ""),
            last_seen=n.get("last_seen", ""),
            sources=n.get("sources", []),
            reports=n.get("reports", []),
            surface_forms=n.get("surface_forms", [name]),
            metadata=n.get("metadata", {}),
        )

    # 2. Migrate Assertions from assertions.jsonl
    logger.info("Loading verified assertions from %s...", ASSERTIONS_FILE)
    assertion_count = 0
    with ASSERTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            assertion_id = a.get("assertion_id") or f"A{assertion_count + 1:06d}"
            
            # Subject
            subj_name = a.get("subject", "UNKNOWN")
            subj_cat = a.get("subject_category", "Entity")
            subj_id = a.get("subject_id") or f"{subj_cat}::{subj_name}"
            
            # Object
            obj_name = a.get("object", "UNKNOWN")
            obj_cat = a.get("object_category", "Entity")
            obj_id = a.get("object_id") or f"{obj_cat}::{obj_name}"
            
            predicate = a.get("predicate", "RELATED_TO")
            evidence = a.get("evidence_sentence") or a.get("claim_text", "")
            conf = float(a.get("confidence", 0.85))
            polarity = a.get("polarity", "positive")
            trigger = a.get("trigger", "")
            report_id = a.get("report_id", "")
            source = a.get("source") or a.get("source_vendor", "")
            pub_date = a.get("publication_date", "")
            s_id = a.get("sentence_id") or a.get("segment_id", "")
            p_id = a.get("paragraph_id", "")

            db.insert_relationship(
                rel_id=assertion_id,
                subject_id=subj_id,
                subject_name=subj_name,
                subject_category=subj_cat,
                predicate=predicate,
                object_id=obj_id,
                object_name=obj_name,
                object_category=obj_cat,
                evidence_text=evidence,
                confidence=conf,
                polarity=polarity,
                trigger=trigger,
                sentence_id=s_id,
                paragraph_id=p_id,
                report_id=report_id,
                source_vendor=source,
                publication_date=pub_date,
                extraction_method=a.get("extraction_method", "rule_based"),
            )
            assertion_count += 1

    stats = db.get_stats()
    logger.info("Migration complete!")
    logger.info("Total Entities in DB: %d", stats["total_entities"])
    logger.info("Total Relationships in DB: %d", stats["total_relationships"])
    logger.info("Category breakdown: %s", stats["category_distribution"])
    logger.info("Predicate breakdown: %s", stats["predicate_distribution"])


if __name__ == "__main__":
    migrate()
