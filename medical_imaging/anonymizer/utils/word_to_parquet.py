"""Split a Word .doc export of patient cases into a parquet file (one row per case).

Each case starts with a header line like:
    PAC ID: 1 HOSP: 11.01.25 - 12.01.25
"""

import argparse
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

CASE_HEADER_RE = re.compile(r"^\s*PAC ID:\s*(\d+)\s+HOSP:\s*(.*?)\s*$")
DATE_RE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})")


def doc_to_text(doc_path: Path) -> str:
    """Convert a binary .doc to plain text using LibreOffice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "txt:Text",
                "--outdir",
                tmpdir,
                str(doc_path),
            ],
            check=True,
            capture_output=True,
        )
        txt_path = Path(tmpdir) / (doc_path.stem + ".txt")
        return txt_path.read_text(encoding="utf-8").replace("\ufeff", "")


def parse_date(match: re.Match) -> date:
    day, month, year = (int(g) for g in match.groups())
    if year < 100:
        year += 2000
    return date(year, month, day)


def split_cases(text: str) -> pd.DataFrame:
    rows = []
    current = None
    for line in text.splitlines():
        header = CASE_HEADER_RE.match(line)
        if header:
            if current is not None:
                rows.append(current)
            dates = list(DATE_RE.finditer(header.group(2)))
            current = {
                "pac_id": int(header.group(1)),
                "hosp_start": parse_date(dates[0]) if len(dates) > 0 else None,
                "hosp_end": parse_date(dates[1]) if len(dates) > 1 else None,
                "text": [],
            }
        elif current is not None:
            current["text"].append(line)
    if current is not None:
        rows.append(current)

    for row in rows:
        row["text"] = "\n".join(row["text"]).strip()
    df = pd.DataFrame(rows)
    df["hosp_start"] = pd.to_datetime(df["hosp_start"])
    df["hosp_end"] = pd.to_datetime(df["hosp_end"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the .doc export")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output parquet path (default: <input>.parquet)",
    )
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".parquet")
    df = split_cases(doc_to_text(args.input))
    df.to_parquet(output, index=False)
    print(f"Wrote {len(df)} cases to {output}")


if __name__ == "__main__":
    main()
