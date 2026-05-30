from markitdown import MarkItDown


def run_markitdown(input_path: str, output_path: str) -> str:
    """Convert a file to Markdown using the markitdown Python API."""
    md = MarkItDown()
    result = md.convert(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    return output_path
