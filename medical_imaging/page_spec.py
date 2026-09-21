"""Parsing of user-entered page selections such as ``"1, 3-5, 8"``.

Shared by the FastAPI server (to pick which PDF pages to OCR) and the UI (to
validate the input before uploading).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def parse_page_spec(spec: str | None) -> list[int] | None:
    """Turn ``"1,3-5"`` into ``[1, 3, 4, 5]``.

    Returns ``None`` when *spec* is empty, ``"all"`` or ``"default"``, meaning
    every page. Page numbers are 1-based, returned sorted and de-duplicated.
    Raises :class:`ValueError` on malformed input.
    """
    if spec is None or not spec.strip() or spec.strip().lower() in {"all", "default", "*"}:
        return None

    pages: set[int] = set()
    for part in spec.split(","):
        if not part.strip():
            continue
        m = _TOKEN.match(part)
        if not m:
            raise ValueError(f"Invalid page selection: {part.strip()!r}")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if start < 1 or end < 1:
            raise ValueError("Page numbers start at 1")
        if end < start:
            raise ValueError(f"Invalid range: {start}-{end}")
        pages.update(range(start, end + 1))

    if not pages:
        return None
    return sorted(pages)


def select_pages(items: list, page_numbers: list[int] | None) -> tuple[list, list[int]]:
    """Return the chosen *items* (1-based page numbers) and the numbers actually used.

    Out-of-range pages are silently skipped; if nothing is left, ``ValueError``
    is raised so the caller can tell the user.
    """
    if page_numbers is None:
        return list(items), list(range(1, len(items) + 1))
    chosen = [n for n in page_numbers if 1 <= n <= len(items)]
    if not chosen:
        raise ValueError(
            f"Requested pages {page_numbers} are out of range; document has {len(items)} page(s)"
        )
    return [items[n - 1] for n in chosen], chosen
