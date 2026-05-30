import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types import Document


def parse_with_docling(file_path: str) -> dict:
    """
    PARSE-01: Parse a file with Docling and extract structure.
    Returns a dict with title, sections, tables, figures, references.
    """
    try:
        converter = DocumentConverter()
        doc: Document = converter.convert(file_path).document

        # Extract title
        title = doc.title or "Untitled"

        # Extract sections with hierarchy
        sections = []
        for node in doc.iter_all_nodes():
            if hasattr(node, "level") and node.level is not None:
                sections.append({
                    "level": node.level,
                    "text": node.export_to_markdown() if hasattr(node, "export_to_markdown") else str(node),
                })

        # Extract tables
        tables = []
        for table in doc.tables or []:
            tables.append({
                "caption": table.caption or "Table",
                "content": str(table),
            })

        # Extract figures
        figures = []
        for figure in doc.figures or []:
            figures.append({
                "caption": figure.caption or "Figure",
                "content": str(figure),
            })

        # Extract references
        references = []
        for ref in doc.bibliography or []:
            references.append({
                "key": getattr(ref, "key", ""),
                "title": getattr(ref, "title", ""),
                "authors": getattr(ref, "authors", []),
            })

        # PARSE-02: Check section count
        section_count = len(sections)

        return {
            "title": title,
            "sections": sections,
            "section_count": section_count,
            "tables": tables,
            "figures": figures,
            "references": references,
            "parse_status": "success",
        }
    except Exception as e:
        # PARSE-03: Handle parsing failures
        return {
            "title": "Error",
            "sections": [],
            "section_count": 0,
            "tables": [],
            "figures": [],
            "references": [],
            "parse_status": "failed",
            "error": str(e),
        }


def save_docling_json(docling_output: dict, output_path: str) -> str:
    """Save Docling extraction result as JSON."""
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docling_output, f, indent=2, default=str)
    return str(path)
