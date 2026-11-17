"""Build hand-written, entity-tagged notes modelled on the real export.

The notes live in ``data/templated_notes/*.txt``. Each note starts with a
header line and is followed by the tagged text::

    @@ f=1 dob=1947-03-12 adm=2025-01-11 dis=2025-01-15 age=77 labs=inf,ren
    <AGE>77letá</AGE> pacientka ... RČ <ID>{RC}</ID> ... {LABS}

Header keys (all optional): ``f`` (1 = female), ``dob``/``adm``/``dis`` (ISO
dates), ``age`` (asserted against dob/adm), ``labs`` (profiles for the
generated lab block: norm, inf, ren, anem, card, hep, dm), ``fmt`` (A/B lab
layout). ``{RC}`` is replaced by a valid rodné číslo for dob/sex and ``{LABS}``
by a lab block in the layout of the real export (lab dates are not PII).

The result has the schema of ``synthetic_notes.parquet``; rows get
``source="synthetic"`` and ids prefixed ``T`` so they can be told apart. With
``--append-to`` they are merged into the main dataset (idempotent: earlier
``T`` rows are replaced).
"""

import argparse
import random
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from medical_imaging.anonymizer.create_dataset import split_by_patient, strip_tags
from medical_imaging.anonymizer.dataset import split_words
from medical_imaging.anonymizer.prompts import ENTITY_LABELS, _rodne_cislo

_ANY_TAG = re.compile(r"</?([A-Za-z_]+)>")
MAX_WIDTH = 12  # words; matches dataset.Dataset default

# (code, unit, low, high, decimals) — reference-ish ranges
BIO = [
    ("MO", "mmol/L", 2.8, 8.0, 1), ("KR", "umol/l", 48, 104, 0), ("NA", "mmol/L", 136, 145, 0),
    ("KA", "mmol/L", 3.6, 5.1, 2), ("CL", "mmol/L", 98, 108, 0), ("BI", "umol/l", 5, 21, 1),
    ("AL", "ukat/l", 0.12, 0.7, 2), ("AS", "ukat/l", 0.15, 0.65, 2), ("GM", "ukat/l", 0.15, 0.9, 2),
    ("AF", "ukat/l", 0.6, 2.0, 2), ("GL", "mmol/L", 4.1, 6.4, 2), ("CRP", "mg/L", 0.3, 7.5, 1),
]
HEM = [
    ("WBC", "x10^9/L", 4.2, 9.8, 1), ("RBC", "x10^12/L", 3.9, 5.4, 1), ("HGB", "g/L", 122, 158, 0),
    ("HCT", "%", 36, 47, 1), ("MCV", "fL", 82, 98, 1), ("MCH", "pg", 27, 33, 1),
    ("MCHC", "g/L", 318, 356, 0), ("PLT", "x10^9/L", 160, 390, 0), ("PDW", "fL", 9.5, 14.5, 1),
]
DIFF = [("NE", 45, 72), ("LY", 18, 40), ("MON", 4, 10), ("EO", 0.5, 4), ("BA", 0.1, 0.9)]
OVERRIDES = {
    "inf": {"CRP": (55, 290), "WBC": (12, 23)},
    "ren": {"KR": (160, 460), "MO": (11, 31), "KA": (4.6, 6.1)},
    "anem": {"HGB": (62, 96), "RBC": (2.3, 3.4), "HCT": (20, 30)},
    "hep": {"BI": (38, 190), "AL": (1.2, 6.5), "AS": (1.1, 7.0), "GM": (1.5, 9.0), "AF": (2.2, 6.0)},
    "dm": {"GL": (11, 24)},
    "card": {},
    "norm": {},
}


def _val(rng, lo, hi, dec):
    v = round(rng.uniform(lo, hi), dec)
    return f"{v:.{max(dec, 1)}f}" if dec else f"{v:.1f}"


def _panel(rng, table, profiles, day_idx):
    cells = []
    for code, unit, lo, hi, dec in table:
        if code not in ("MO", "KR", "NA", "KA", "CL", "CRP", "WBC", "RBC", "HGB", "HCT",
                        "MCV", "MCH", "MCHC", "PLT") and rng.random() < 0.3:
            continue
        for p in profiles:
            if code in OVERRIDES[p]:
                lo, hi = OVERRIDES[p][code]
                if code in ("CRP", "WBC") and day_idx:  # inflammation recedes
                    hi = lo + (hi - lo) / (1 + day_idx)
        cells.append(f"{code} {_val(rng, lo, hi, dec)} {unit}")
        if code == "KR":
            kr = float(cells[-1].split()[1])
            cells.append(f"GFR {min(2.2, round(95 / kr, 2))} ml/s/1,73 m2")
    return cells


def lab_block(rng: random.Random, profiles: list[str], adm: date, dis: date, fmt: str) -> str:
    span = max(0, (dis - adm).days)
    n_days = min(span + 1, rng.choice([1, 1, 2, 2, 3, 4]))
    days = sorted(rng.sample(range(span + 1), n_days))
    days[0] = 0
    days = sorted(set(days))

    def render(title, panels):
        if fmt == "A":
            body = " ".join(f"{d:%d.%m.%Y}: {' | '.join(c)}" for d, c in panels)
            return f"{title}: {body}"
        first, *rest = panels
        out = f"{title}:{first[0]:%d.%m.%Y}:{' | '.join(first[1])} "
        for d, c in rest:
            out += f"\n{d:%d.%m.%Y}:\n{' | '.join(c)} "
        return out.rstrip() if rng.random() < 0.5 else out

    bio, hem = [], []
    for i, off in enumerate(days):
        d = adm + timedelta(days=off)
        cells = _panel(rng, BIO, profiles, i)
        if "card" in profiles and i == 0:
            cells += [f"BNP {_val(rng, 900, 9000, 0)} ng/L", f"TNI {_val(rng, 20, 900, 0)} ng/L"]
        if rng.random() < 0.6:
            cells += ["HEMOL 0.0", f"ICTER {'1.0 -' if 'hep' in profiles else '0.0'}", "LIPEM 0.0"]
        if i == 0 and rng.random() < 0.35:
            cells += [
                f"UCOL {rng.choice(['žlutá', 'sv. žlutá', 'tm. žlutá', 'jantarová'])}",
                f"UCLA {rng.choice(['bez zákalu', 'mírně zak.', 'zakaleno'])}",
                f"UPH {rng.choice(['5.0', '5.5', '6.0', '6.5', '7.0'])} -",
                f"USH 1.0{rng.randint(5, 30):02d} -", "UBILK 0.0", "UGK-S 0.0", "UAC 0.0",
                f"ULEU {_val(rng, 0, 400 if 'inf' in profiles else 20, 0)} 10^6/L",
                f"UERY {_val(rng, 0, 30, 0)} 10^6/L",
            ]
        bio.append((d, cells + ["REPT_NTS"]))
        if i == 0 or rng.random() < 0.6:
            cells = _panel(rng, HEM, profiles, i) + ["NRBC 0.0", "NRBC% 0.0"]
            if rng.random() < 0.4:
                cells += [f"{c} {_val(rng, lo, hi, 1)} %" for c, lo, hi in DIFF]
            hem.append((d, cells + ["REPT_NTS"]))
    return render("Biochemie", bio) + "\n" + render("Hematologie", hem)


def parse_notes(paths: list[Path]) -> list[tuple[dict, str]]:
    notes = []
    for path in paths:
        for chunk in re.split(r"^@@", path.read_text(encoding="utf-8"), flags=re.M)[1:]:
            header, _, body = chunk.partition("\n")
            meta = dict(kv.split("=", 1) for kv in header.split())
            notes.append((meta, body.strip("\n")))
    return notes


def build(notes: list[tuple[dict, str]], seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    labels = set(ENTITY_LABELS)
    rows, problems = [], []
    for i, (meta, body) in enumerate(notes, 1):
        tag = f"note {i} ({body[:40]!r})"
        female = meta.get("f") == "1"
        dob = date.fromisoformat(meta["dob"]) if "dob" in meta else None
        adm = date.fromisoformat(meta["adm"]) if "adm" in meta else None
        dis = date.fromisoformat(meta["dis"]) if "dis" in meta else adm

        if "age" in meta and dob and adm:
            age = adm.year - dob.year - ((adm.month, adm.day) < (dob.month, dob.day))
            if age != int(meta["age"]):
                problems.append(f"{tag}: age {meta['age']} != {age} from dob/adm")
        if "{RC}" in body:
            if not dob:
                problems.append(f"{tag}: {{RC}} without dob")
                continue
            rc = _rodne_cislo(dob.year, dob.month, dob.day, female, rng)
            if rng.random() < 0.25:
                rc = rc.replace("/", "")
            body = body.replace("{RC}", rc)
        if "{LABS}" in body:
            if not adm:
                problems.append(f"{tag}: {{LABS}} without adm")
                continue
            profiles = meta.get("labs", "norm").split(",")
            fmt = meta.get("fmt") or rng.choice("AAB")
            body = body.replace("{LABS}", lab_block(rng, profiles, adm, dis, fmt))
        if "{" in body or "}" in body:
            problems.append(f"{tag}: unresolved placeholder")

        # Tag sanity: known labels, strictly alternating open/close, no nesting.
        open_label = None
        for m in _ANY_TAG.finditer(body):
            closing, label = m.group(0).startswith("</"), m.group(1)
            if label not in labels:
                problems.append(f"{tag}: unknown tag {m.group(0)}")
            elif closing != (open_label is not None) or (closing and label != open_label):
                problems.append(f"{tag}: unbalanced tag {m.group(0)} at {m.start()}")
            open_label = None if closing else label
        if open_label:
            problems.append(f"{tag}: unclosed <{open_label}>")

        text, entities = strip_tags(body)
        for e in entities:
            if e["text"] != e["text"].strip():
                problems.append(f"{tag}: whitespace at edge of {e['text']!r}")
            if len(split_words(e["text"])) > MAX_WIDTH:
                problems.append(f"{tag}: entity wider than {MAX_WIDTH} words: {e['text']!r}")
            assert text[e["start"] : e["end"]] == e["text"]
        pid = f"T{i:04d}"
        rows.append(
            {
                "doc_id": f"{pid}_d0",
                "patient_id": pid,
                "source": "synthetic",
                "text": text,
                "entities": entities,
                "n_entities": len(entities),
                "raw_tagged": body,
            }
        )
    if problems:
        raise SystemExit("\n".join(problems))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--notes-dir", type=Path, default=Path("data/templated_notes"))
    p.add_argument("--output", type=Path, default=Path("data/templated_notes.parquet"))
    p.add_argument("--append-to", type=Path, default=None,
                   help="Existing dataset parquet to merge the new rows into")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = build(parse_notes(sorted(args.notes_dir.glob("*.txt"))), args.seed)
    df["split"] = split_by_patient(df, args.test_size, args.seed)
    df.to_parquet(args.output, index=False)
    counts = pd.Series([e["label"] for es in df.entities for e in es]).value_counts().to_dict()
    print(f"Wrote {len(df)} notes to {args.output}; split {df.split.value_counts().to_dict()}")
    print(f"  entities: {counts}")
    print(f"  chars/note: median {int(df.text.str.len().median())}, "
          f"entities/note: median {int(df.n_entities.median())}")

    if args.append_to:
        base = pd.read_parquet(args.append_to)
        base = base[~base["patient_id"].str.match(r"T\d{4}$")]
        base["entities"] = base["entities"].map(lambda es: [dict(e) for e in es])
        merged = pd.concat([base, df], ignore_index=True)
        merged.to_parquet(args.append_to, index=False)
        print(f"Merged into {args.append_to}: {len(base)} -> {len(merged)} rows")


if __name__ == "__main__":
    main()
