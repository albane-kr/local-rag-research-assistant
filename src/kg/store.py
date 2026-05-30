"""Simple in-memory knowledge graph store."""
from typing import Optional


class KGStore:
    """Simple knowledge graph backed by dicts."""

    def __init__(self):
        self.nodes = {}  # id -> node dict
        self.edges = []  # list of edge dicts

    def add_nodes(self, nodes: list[dict]):
        """KG-01: Add nodes to the graph."""
        for node in nodes:
            node_id = node["id"]
            self.nodes[node_id] = node

    def add_edges(self, edges: list[dict]):
        """KG-02: Add edges to the graph."""
        self.edges.extend(edges)

    def deactivate_old_versions(self, resource_id: str, current_version: int):
        """KG-03: Mark old version nodes/edges as inactive."""
        # Remove nodes from old versions
        to_delete = [
            nid for nid, node in self.nodes.items()
            if node.get("resource_id") == resource_id
            and node.get("version") != current_version
        ]
        for nid in to_delete:
            del self.nodes[nid]

        # Remove edges from old versions
        self.edges = [
            e for e in self.edges
            if not (e.get("resource_id") == resource_id and e.get("version") != current_version)
        ]

    def get_context_for_query(
        self,
        query: str,
        resource_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """
        Build KG context string for augmenting prompts.
        Returns formatted text summarizing relevant nodes and relations.
        """
        if not self.nodes:
            return ""

        # Simple approach: list recent nodes and their relations
        context_parts = []

        # Get nodes for this resource
        relevant_nodes = [
            n for n in self.nodes.values()
            if resource_id is None or n.get("resource_id") == resource_id
        ]

        if not relevant_nodes:
            return ""

        # Group by type
        docs = [n for n in relevant_nodes if n.get("type") == "document"]
        sections = [n for n in relevant_nodes if n.get("type") == "section"]
        refs = [n for n in relevant_nodes if n.get("type") == "reference"]

        if docs:
            context_parts.append(f"Documents: {', '.join(d['label'] for d in docs[:3])}")

        if refs:
            context_parts.append(f"Key references: {', '.join(r['label'] for r in refs[:3])}")

        # Find relations mentioning key entities
        relevant_edges = [
            e for e in self.edges
            if resource_id is None or e.get("resource_id") == resource_id
        ]

        if relevant_edges:
            relations_text = []
            for edge in relevant_edges[:5]:
                source = self.nodes.get(edge["source"], {}).get("label", "")
                target = self.nodes.get(edge["target"], {}).get("label", "")
                if source and target:
                    relations_text.append(f"{source} {edge['relation']} {target}")
            if relations_text:
                context_parts.append(f"Relations: {'; '.join(relations_text)}")

        return "\n".join(context_parts) if context_parts else ""

    def get_stats(self, resource_id: Optional[str] = None) -> dict:
        """Get graph statistics."""
        if resource_id:
            nodes = [n for n in self.nodes.values() if n.get("resource_id") == resource_id]
            edges = [e for e in self.edges if e.get("resource_id") == resource_id]
        else:
            nodes = list(self.nodes.values())
            edges = self.edges

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": list(set(n.get("type") for n in nodes if n.get("type"))),
        }
