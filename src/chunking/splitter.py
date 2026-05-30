import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    resource_id: str
    version: int
    chunk_index: int
    heading: str
    is_latest: int = 1  # 1=active, 0=superseded; int for Chroma metadata compatibility


def split_by_headings(
    markdown_text: str,
    resource_id: str,
    version: int,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """
    Split markdown into Chunk objects:
    1. Primary split at heading boundaries (H1–H6).
    2. Sub-split by paragraph when a section exceeds max_chars.
    3. Append the tail of the previous chunk (overlap_chars) to each chunk
       so queries near section boundaries still find relevant context.

    max_chars=1000 keeps chunks within all-MiniLM-L6-v2's 256-token limit
    (~4 chars/token). overlap_chars=100 ≈ 25 tokens of cross-chunk context.
    """
    heading_re = re.compile(r"^(#{1,6} .+)$", re.MULTILINE)
    parts = heading_re.split(markdown_text)

    # Pair each body block with its nearest preceding heading
    sections: list[tuple[str, str]] = []
    current_heading = ""
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if heading_re.match(stripped):
            current_heading = stripped
        else:
            sections.append((current_heading, stripped))

    # Sub-split sections that exceed max_chars, breaking at paragraph boundaries
    raw: list[tuple[str, str]] = []
    for heading, body in sections:
        if len(body) <= max_chars:
            raw.append((heading, body))
            continue
        paras = re.split(r"\n\n+", body)
        bucket = ""
        for para in paras:
            if bucket and len(bucket) + len(para) + 2 > max_chars:
                raw.append((heading, bucket.strip()))
                bucket = para
            else:
                bucket = (bucket + "\n\n" + para).lstrip() if bucket else para
        if bucket:
            raw.append((heading, bucket.strip()))

    chunks: list[Chunk] = []
    for i, (heading, body) in enumerate(raw):
        text = f"{heading}\n\n{body}" if heading else body
        if i > 0 and overlap_chars > 0:
            tail = chunks[-1].text[-overlap_chars:]
            text = tail + "\n\n" + text
        chunks.append(Chunk(
            chunk_id=f"{resource_id}_v{version}_{i}",
            text=text,
            resource_id=resource_id,
            version=version,
            chunk_index=i,
            heading=heading,
        ))

    return chunks
