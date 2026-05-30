"""Extract entities and relations from Docling JSON for KG construction."""
from src.kg.store import KGStore


def extract_and_populate_kg(
    docling_output: dict,
    resource_id: str,
    version: int,
) -> KGStore:
    """
    KG-01/02/03: Extract entities and relations from Docling output.
    Returns populated KGStore with nodes and edges linked to resource_id + version.
    """
    kg = KGStore()

    # Extract document node
    doc_node = {
        "id": f"{resource_id}_v{version}_doc",
        "type": "document",
        "label": docling_output.get("title", "Untitled"),
        "resource_id": resource_id,
        "version": version,
    }
    kg.add_nodes([doc_node])

    # Extract section nodes
    section_nodes = []
    for i, section in enumerate(docling_output.get("sections", [])[:10]):  # Limit to 10
        section_node = {
            "id": f"{resource_id}_v{version}_section_{i}",
            "type": "section",
            "label": section.get("text", "Section")[:100],  # Truncate for label
            "level": section.get("level", 0),
            "resource_id": resource_id,
            "version": version,
        }
        section_nodes.append(section_node)

    kg.add_nodes(section_nodes)

    # Add edges from document to sections
    doc_section_edges = [
        {
            "source": doc_node["id"],
            "target": sn["id"],
            "relation": "contains",
            "resource_id": resource_id,
            "version": version,
        }
        for sn in section_nodes
    ]
    kg.add_edges(doc_section_edges)

    # Extract reference nodes (KG-01: authors + concepts)
    ref_nodes = []
    for i, ref in enumerate(docling_output.get("references", [])[:10]):
        ref_node = {
            "id": f"{resource_id}_v{version}_ref_{i}",
            "type": "reference",
            "label": ref.get("title", "Untitled"),
            "authors": ref.get("authors", []),
            "resource_id": resource_id,
            "version": version,
        }
        ref_nodes.append(ref_node)

    kg.add_nodes(ref_nodes)

    # Add edges from document to references (KG-02: cites relation)
    doc_ref_edges = [
        {
            "source": doc_node["id"],
            "target": rn["id"],
            "relation": "cites",
            "resource_id": resource_id,
            "version": version,
        }
        for rn in ref_nodes
    ]
    kg.add_edges(doc_ref_edges)

    # Extract table nodes
    table_nodes = []
    for i, table in enumerate(docling_output.get("tables", [])[:5]):
        table_node = {
            "id": f"{resource_id}_v{version}_table_{i}",
            "type": "table",
            "label": table.get("caption", "Table"),
            "resource_id": resource_id,
            "version": version,
        }
        table_nodes.append(table_node)

    kg.add_nodes(table_nodes)

    # Add edges from sections to tables
    if section_nodes and table_nodes:
        section_table_edges = [
            {
                "source": section_nodes[0]["id"],
                "target": tn["id"],
                "relation": "contains",
                "resource_id": resource_id,
                "version": version,
            }
            for tn in table_nodes
        ]
        kg.add_edges(section_table_edges)

    return kg
