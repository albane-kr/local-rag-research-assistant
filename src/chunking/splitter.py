import re


def split_by_headings(markdown_text: str):
    """Naive splitter that groups headings with following text."""
    parts = re.split(r'(^#{1,6} .*$)', markdown_text, flags=re.M)
    chunks = []
    current = ""
    for p in parts:
        if p.strip().startswith("#"):
            if current:
                chunks.append(current.strip())
            current = p
        else:
            current += p
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]
