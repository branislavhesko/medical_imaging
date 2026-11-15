"""Generate a synthetic, entity-tagged clinical-note dataset with vLLM.

The LLM writes Czech clinical notes with PII marked by inline XML tags
(``<NAME>Jan Novák</NAME>``); stripping the tags yields the clean text plus
exact character offsets, so the output can train a NER/anonymization model.

Real notes (``data/Export.parquet``) are appended as ``source="real"`` and are
always assigned to the test split — annotate them in-house and never train on
them (synthetic test sets overestimate performance). Splitting is grouped by
``patient_id`` so a patient's identifiers can't leak across train/test.
"""

import argparse
import random
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .prompts import DOC_TYPES, ENTITY_LABELS, build_messages, sample_patient

_PAIR_RE = re.compile(r"<([A-Z_]+)>(.*?)</\1>", re.DOTALL)
_TAG_RE = re.compile(r"</?[A-Z_]+>")
_LABELS = set(ENTITY_LABELS)


def strip_tags(tagged: str) -> tuple[str, list[dict]]:
    """Remove inline XML tags, returning (clean_text, entities).

    Entities are dicts {start, end, label, text} with offsets into clean_text.
    Unknown labels and malformed/stray tags are dropped but their inner text is
    preserved.
    """
    parts, entities, pos, last = [], [], 0, 0
    for m in _PAIR_RE.finditer(tagged):
        segment = _TAG_RE.sub("", tagged[last : m.start()])
        parts.append(segment)
        pos += len(segment)
        surface = _TAG_RE.sub("", m.group(2))
        if m.group(1) in _LABELS and surface.strip():
            entities.append(
                {"start": pos, "end": pos + len(surface), "label": m.group(1), "text": surface}
            )
        parts.append(surface)
        pos += len(surface)
        last = m.end()
    parts.append(_TAG_RE.sub("", tagged[last:]))
    return "".join(parts), entities


def generate(
    n_patients: int,
    docs_per_patient: int,
    model: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> pd.DataFrame:
    from vllm import LLM, SamplingParams

    rng = random.Random(seed)
    seeds, convs = [], []
    for i in range(n_patients):
        patient = sample_patient(i + 1, rng)
        for d in range(docs_per_patient):
            s = patient if d == 0 else replace(patient, doc_type=rng.choice(DOC_TYPES))
            seeds.append(s)
            convs.append(build_messages(s, rng))

    # Per-request sampling keeps docs reproducible while varying decoding style.
    sampling = [
        SamplingParams(
            temperature=max(0.3, min(1.2, rng.gauss(temperature, 0.15))),
            top_p=rng.uniform(max(0.8, top_p - 0.1), top_p),
            presence_penalty=rng.choice([0.0, 0.0, 0.3, 0.6]),
            frequency_penalty=rng.choice([0.0, 0.0, 0.2, 0.4]),
            max_tokens=max_tokens,
            seed=seed + i,
        )
        for i in range(len(convs))
    ]

    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    outputs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})

    rows = []
    for i, (s, out) in enumerate(zip(seeds, outputs)):
        tagged = out.outputs[0].text.strip()
        clean, entities = strip_tags(tagged)
        rows.append(
            {
                "doc_id": f"{s.patient_id}_d{i}",
                "patient_id": s.patient_id,
                "source": "synthetic",
                "text": clean,
                "entities": entities,
                "n_entities": len(entities),
                "raw_tagged": tagged,
            }
        )
    return pd.DataFrame(rows)


def split_by_patient(df: pd.DataFrame, test_size: float, seed: int) -> pd.Series:
    """Assign 'train'/'test' so no patient_id appears in both splits."""
    from sklearn.model_selection import GroupShuffleSplit

    split = pd.Series("train", index=df.index)
    if 0 < test_size < 1 and df["patient_id"].nunique() > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        test_idx = next(gss.split(df, groups=df["patient_id"]))[1]
        split.iloc[test_idx] = "test"
    return split


def load_real_notes(path: Path) -> pd.DataFrame:
    """Real notes are test-only; entities stay empty until annotated in-house."""
    df = pd.read_parquet(path)
    return pd.DataFrame(
        {
            "doc_id": [f"real_{i}" for i in df.index],
            "patient_id": [f"real_pac_{p}" for p in df["pac_id"]],
            "source": "real",
            "text": df["text"],
            "entities": [[] for _ in range(len(df))],
            "n_entities": 0,
            "raw_tagged": None,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--n-patients", type=int, default=250)
    p.add_argument("--docs-per-patient", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("data/synthetic_notes.parquet"))
    p.add_argument("--real-notes", type=Path, default=None,
                   help="Parquet of real notes (pac_id, text); appended as test-only rows")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = p.parse_args()

    df = generate(
        args.n_patients,
        args.docs_per_patient,
        args.model,
        args.seed,
        args.temperature,
        args.top_p,
        args.max_tokens,
        args.max_model_len,
        args.gpu_memory_utilization,
    )
    df["split"] = split_by_patient(df, args.test_size, args.seed)

    if args.real_notes and args.real_notes.exists():
        real = load_real_notes(args.real_notes)
        real["split"] = "test"
        df = pd.concat([df, real], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    bad = df[(df.source == "synthetic") & (df.n_entities < 3)]
    print(
        f"Wrote {len(df)} docs ({(df.source == 'synthetic').sum()} synthetic, "
        f"{(df.source == 'real').sum()} real) to {args.output}\n"
        f"  split: {df.split.value_counts().to_dict()}, "
        f"synthetic docs with <3 entities: {len(bad)}"
    )


if __name__ == "__main__":
    main()
