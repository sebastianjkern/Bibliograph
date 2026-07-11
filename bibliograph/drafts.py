import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DraftClaim:
    text: str
    source_format: str
    line_start: int
    citation_keys: tuple[str, ...] = ()


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".tex", ".latex"}:
        return "latex"
    if suffix == ".typ":
        return "typst"
    raise ValueError(f"Unsupported draft format: {suffix or 'missing extension'}")


def parse_draft_file(path: str | Path, source_format: str | None = None) -> list[DraftClaim]:
    draft_path = Path(path)
    if not draft_path.is_file():
        raise FileNotFoundError(draft_path)
    return parse_draft(
        draft_path.read_text(encoding="utf-8"),
        source_format or detect_format(draft_path),
    )


def parse_draft(text: str, source_format: str) -> list[DraftClaim]:
    if source_format not in {"latex", "typst"}:
        raise ValueError("source_format must be 'latex' or 'typst'")
    claims: list[DraftClaim] = []
    offset = 0
    for block in re.split(r"\n\s*\n", text):
        line_start = text.count("\n", 0, offset) + 1
        offset += len(block) + 1
        if source_format == "latex":
            citation_keys = _latex_citations(block)
            cleaned = _clean_latex(block)
        else:
            citation_keys = _typst_citations(block)
            cleaned = _clean_typst(block)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned.split()) >= 4:
            claims.append(DraftClaim(cleaned, source_format, line_start, citation_keys))
    return claims


def _latex_citations(text: str) -> tuple[str, ...]:
    keys = re.findall(r"\\(?:cite|citep|citet|parencite|textcite)(?:\w*)?\s*\{([^}]+)\}", text)
    return _unique(key.strip() for group in keys for key in group.split(","))


def _typst_citations(text: str) -> tuple[str, ...]:
    keys = re.findall(r"@([A-Za-z][\w:-]*)|#cite\(\s*<([^>]+)>", text)
    return _unique(key for pair in keys for key in pair if key)


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
