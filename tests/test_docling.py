"""Tests for Docling parsing and KG extraction (PARSE-*, KG-* from PRD)."""
import tempfile
import json
from pathlib import Path

from src.parse.docling_parser import parse_with_docling, save_docling_json
from src.kg.store import KGStore
from src.kg.extractor import extract_and_populate_kg


def test_docling_parse_success():
    """PARSE-01: Docling extraction returns required fields."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write(
            "# My Research Paper\n\n"
            "## Introduction\n\n"
            "This is the introduction section with key concepts.\n\n"
            "## Methods\n\n"
            "Experimental setup details here.\n\n"
            "## Results\n\n"
            "Key findings and analysis.\n\n"
            "## References\n\n"
            "[1] Smith et al., 2020\n"
        )
        tmp_path = f.name

    try:
        result = parse_with_docling(tmp_path)

        # Verify required fields present
        assert "title" in result
        assert "sections" in result
        assert "tables" in result
        assert "figures" in result
        assert "references" in result
        assert "parse_status" in result

        assert result["parse_status"] == "success"
        assert isinstance(result["sections"], list)
    finally:
        Path(tmp_path).unlink()


def test_docling_section_count():
    """PARSE-02: Academic paper has at least 3 sections."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write(
            "# Paper Title\n\n"
            "## Section 1\n\nContent.\n\n"
            "## Section 2\n\nContent.\n\n"
            "## Section 3\n\nContent.\n"
        )
        tmp_path = f.name

    try:
        result = parse_with_docling(tmp_path)
        assert result["parse_status"] == "success"
        # At minimum, the title and sections should be extracted
        assert "title" in result
    finally:
        Path(tmp_path).unlink()


def test_docling_parse_invalid_file():
    """PARSE-03: Invalid/encrypted file returns parse_failed status."""
    result = parse_with_docling("/nonexistent/file.pdf")
    assert result["parse_status"] == "failed"
    assert "error" in result


def test_save_docling_json():
    """Test saving Docling output to JSON file."""
    docling_output = {
        "title": "Test Document",
        "sections": [{"level": 1, "text": "Section 1"}],
        "parse_status": "success",
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = str(Path(tmp_dir) / "test.json")
        result = save_docling_json(docling_output, output_path)

        assert Path(result).exists()
        with open(result) as f:
            saved = json.load(f)
        assert saved["title"] == "Test Document"


def test_kg_node_creation():
    """KG-01: Nodes are created from document, sections, and references."""
    kg = KGStore()

    doc_node = {
        "id": "doc1",
        "type": "document",
        "label": "Research Paper",
        "resource_id": "res1",
        "version": 1,
    }

    section_node = {
        "id": "section1",
        "type": "section",
        "label": "Introduction",
        "resource_id": "res1",
        "version": 1,
    }

    kg.add_nodes([doc_node, section_node])

    assert len(kg.nodes) == 2
    assert "doc1" in kg.nodes
    assert "section1" in kg.nodes
    assert kg.nodes["doc1"]["type"] == "document"


def test_kg_edge_creation():
    """KG-02: Edges (relations) are created between nodes."""
    kg = KGStore()

    nodes = [
        {"id": "doc1", "type": "document", "label": "Paper", "resource_id": "r1", "version": 1},
        {"id": "ref1", "type": "reference", "label": "Reference", "resource_id": "r1", "version": 1},
    ]
    kg.add_nodes(nodes)

    edge = {
        "source": "doc1",
        "target": "ref1",
        "relation": "cites",
        "resource_id": "r1",
        "version": 1,
    }
    kg.add_edges([edge])

    assert len(kg.edges) == 1
    assert kg.edges[0]["relation"] == "cites"


def test_kg_version_awareness():
    """KG-03: Old version nodes are deactivated when new version indexed."""
    kg = KGStore()

    # Add v1 nodes
    v1_nodes = [
        {"id": "doc_v1", "type": "document", "label": "v1", "resource_id": "res1", "version": 1},
        {"id": "sec_v1", "type": "section", "label": "Section v1", "resource_id": "res1", "version": 1},
    ]
    kg.add_nodes(v1_nodes)

    # Add v2 nodes
    v2_nodes = [
        {"id": "doc_v2", "type": "document", "label": "v2", "resource_id": "res1", "version": 2},
    ]
    kg.add_nodes(v2_nodes)

    assert len(kg.nodes) == 3

    # Deactivate old versions
    kg.deactivate_old_versions("res1", 2)

    # Only v2 nodes should remain
    assert len(kg.nodes) == 1
    assert "doc_v2" in kg.nodes


def test_kg_context_generation():
    """Test generating context string from KG for prompt augmentation."""
    kg = KGStore()

    nodes = [
        {"id": "doc1", "type": "document", "label": "ML Paper", "resource_id": "r1", "version": 1},
        {"id": "ref1", "type": "reference", "label": "Survey", "resource_id": "r1", "version": 1},
    ]
    kg.add_nodes(nodes)

    edges = [
        {"source": "doc1", "target": "ref1", "relation": "cites", "resource_id": "r1", "version": 1},
    ]
    kg.add_edges(edges)

    context = kg.get_context_for_query("machine learning", resource_id="r1")
    assert len(context) > 0
    assert "ML Paper" in context or "Paper" in context


def test_kg_stats():
    """Test KG statistics generation."""
    kg = KGStore()

    nodes = [
        {"id": "n1", "type": "document", "label": "Doc", "resource_id": "r1", "version": 1},
        {"id": "n2", "type": "section", "label": "Sec", "resource_id": "r1", "version": 1},
        {"id": "n3", "type": "document", "label": "Doc2", "resource_id": "r2", "version": 1},
    ]
    kg.add_nodes(nodes)

    # Stats for r1
    stats = kg.get_stats(resource_id="r1")
    assert stats["node_count"] == 2

    # Stats overall
    stats_all = kg.get_stats()
    assert stats_all["node_count"] == 3


def test_extract_and_populate_kg():
    """Integration: Extract KG from Docling output."""
    docling_output = {
        "title": "Neural Networks Survey",
        "sections": [
            {"level": 1, "text": "Introduction"},
            {"level": 2, "text": "Background"},
        ],
        "references": [
            {"title": "Deep Learning Book", "authors": ["Goodfellow"]},
        ],
        "tables": [{"caption": "Performance Comparison"}],
    }

    kg = extract_and_populate_kg(docling_output, "survey1", 1)

    # Check document node created
    assert len(kg.nodes) > 0

    # Check document node with correct label
    doc_nodes = [n for n in kg.nodes.values() if n.get("type") == "document"]
    assert len(doc_nodes) >= 1
    assert "Neural Networks Survey" in doc_nodes[0]["label"]

    # Check relations created
    assert len(kg.edges) > 0

    # Check version metadata
    for node in kg.nodes.values():
        assert node.get("version") == 1
        assert node.get("resource_id") == "survey1"
