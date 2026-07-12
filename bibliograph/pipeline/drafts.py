"""Parse supported draft formats into lightweight claim dictionaries."""

import re
from pathlib import Path

from ..domain import Claim


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".tex", ".latex"}:
        return "latex"
    if suffix == ".typ":
        return "typst"
    raise ValueError(f"Unsupported draft format: {suffix or 'missing extension'}")


def parse_draft_file(path: str | Path, source_format: str | None = None) -> list[Claim]:
    draft_path = Path(path)
    if not draft_path.is_file():
        raise FileNotFoundError(draft_path)
    return parse_draft(
        draft_path.read_text(encoding="utf-8"),
        source_format or detect_format(draft_path),
    )


def parse_draft(text: str, source_format: str) -> list[Claim]:
    if source_format not in {"latex", "typst"}:
        raise ValueError("source_format must be 'latex' or 'typst'")
    claims: list[Claim] = []
    offset = 0
    for part in re.split(r"(\n\s*\n)", text):
        if re.fullmatch(r"\n\s*\n", part):
            offset += len(part)
            continue
        block = part
        line_start = text.count("\n", 0, offset) + 1
        offset += len(part)
        citations = _latex_citations(block) if source_format == "latex" else _typst_citations(block)
        cleaned = _clean_latex(block) if source_format == "latex" else _clean_typst(block)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned.split()) >= 4:
            claims.append(
                {
                    "text": cleaned,
                    "source_format": source_format,
                    "line_start": line_start,
                    "citation_keys": citations,
                }
            )
    return claims


def _latex_citations(text: str) -> tuple[str, ...]:
    matches = re.findall(r"\\(?:cite|citep|citet|parencite|textcite)(?:\w*)?\s*\{([^}]+)\}", text)
    return _unique(key.strip() for match in matches for key in match.split(","))


def _typst_citations(text: str) -> tuple[str, ...]:
    matches = re.findall(r"@([A-Za-z][\w:-]*)|#cite\(\s*<([^>]+)>", text)
    return _unique(value for pair in matches for value in pair if value)


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _clean_latex(text: str) -> str:
    text = re.sub(r"(?m)%.*$", "", text)
    text = re.sub(r"\\(?:cite\w*|ref|label)\s*\{[^}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?\s*", "", text)
    return re.sub(r"[{}]", "", text)


def _clean_typst(text: str) -> str:
    text = re.sub(r"(?m)//.*$", "", text)
    text = re.sub(r"#(?:cite|ref)\(\s*<[^>]+>\s*\)", "", text)
    text = re.sub(r"@[A-Za-z][\w:-]*", "", text)
    text = re.sub(r"#(?:[A-Za-z][\w-]*)(?:\([^)]*\))?", "", text)
    return re.sub(r"[*_`~]", "", text)
