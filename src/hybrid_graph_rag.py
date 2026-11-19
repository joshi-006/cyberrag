"""Initializes dense, hybrid and graph retrievers once and reuses them."""
from src.retrieval import StandardRAGRetriever, HybridRAGRetriever, ClaimConditionedRetriever
from src.graph_retrieval import GraphRAGRetriever


class HybridGraphRAG:
    def __init__(
        self,
        data_path="data/processed_segments.jsonl",
        graph_path="data/cti_knowledge_graph_trusted.json",
        assertions_path="data/assertions.jsonl",
        dense_retriever=None,
        graph_retriever=None,
        gate=None,
        max_per_publisher=2,
    ):
        print("Initializing retrieval systems...")
        self.dense = dense_retriever or StandardRAGRetriever(data_path=data_path)
        self.hybrid = HybridRAGRetriever(self.dense)
        self.graph = graph_retriever or GraphRAGRetriever(
            graph_path=graph_path, assertions_path=assertions_path
        )
        self.claim_conditioned = ClaimConditionedRetriever(
            self.hybrid, gate=gate, max_per_publisher=max_per_publisher
        )
        print("All retrieval systems are ready!")

    def retrieve(self, query, top_k=5):
        return {
            "dense": self.dense.retrieve(query, top_k=top_k),
            "hybrid": self.hybrid.retrieve(query, top_k=top_k),
            "graph": self.graph.retrieve(query, top_k=top_k),
        }

    def retrieve_for_claim(
        self,
        subject: str,
        predicate: str,
        obj: str,
        natural_query: str = None,
        top_k: int = 5,
    ):
        return {
            "claim_conditioned": self.claim_conditioned.retrieve_for_claim(
                subject=subject,
                predicate=predicate,
                obj=obj,
                natural_query=natural_query,
                top_k=top_k,
            ),
            "graph": self.graph.retrieve(natural_query or f"{subject} {predicate} {obj}", top_k=top_k),
        }

