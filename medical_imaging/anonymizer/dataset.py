"""Span-NER dataset over the synthetic notes, matching ``detector.SpanNER``.

The model consumes subword tokens, pools one vector per *word* (the first
subword, via ``word_pos``) and scores every span of up to ``max_width`` words.
Targets are therefore word-level spans, built here from the parquet's
character-offset entities.

Words are ``\\w+`` runs or single punctuation marks, so entity boundaries such
as ``Hradiště,`` always fall between words.
"""

import logging
import re

import pandas as pd
import torch
import torch.utils.data as tdata
from transformers import AutoTokenizer

from medical_imaging.anonymizer.prompts import ENTITY_LABELS

logger = logging.getLogger(__name__)

LABELS = list(ENTITY_LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def split_words(text: str) -> list[tuple[int, int]]:
    """Character spans [start, end) of each word in ``text``."""
    return [m.span() for m in _WORD_RE.finditer(text)]


def entity_to_word_span(words: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    """Inclusive (first_word, last_word) covering the character span [start, end)."""
    covered = [i for i, (ws, we) in enumerate(words) if ws < end and we > start]
    return (covered[0], covered[-1]) if covered else None


class Dataset(tdata.Dataset):
    """One item per token window of a note.

    Long notes are split into overlapping windows of ``max_length`` tokens
    (``stride`` tokens of overlap). Each window keeps only the words whose
    first subword lies inside it, and only the entities whose words all lie
    inside it; with a large enough stride every entity appears whole in at
    least one window. ``self.doc_index[i]`` is the ``data`` row of window ``i``.

    Item keys:
        input_ids, attention_mask: [T] token ids / mask
        word_pos:   [N] index of each word's first subword within the window
        word_chars: [N, 2] character span of each word in the original text
        spans:      [S, 3] (first_word, last_word, label_id), word indices
                    relative to the window, last_word inclusive
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        stride: int = 128,
        max_width: int = 12,
    ):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for offset mapping")
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.max_width = max_width
        self.too_wide = 0  # entities longer than max_width words (unlearnable)

        self.items, self.doc_index = [], []
        for idx in range(len(self.data)):
            for item in self._encode(idx):
                self.items.append(item)
                self.doc_index.append(idx)
        if self.too_wide:
            logger.warning("%d entities exceed max_width=%d words and were dropped",
                           self.too_wide, max_width)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def _encode(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        words = split_words(text)

        # Entities as inclusive word spans over the whole note.
        doc_spans = []
        for ent in row["entities"]:
            if ent["label"] not in LABEL2ID:
                continue
            ws = entity_to_word_span(words, int(ent["start"]), int(ent["end"]))
            if ws is None:
                continue
            if ws[1] - ws[0] + 1 > self.max_width:
                self.too_wide += 1
                continue
            doc_spans.append((ws[0], ws[1], LABEL2ID[ent["label"]]))

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        for w, offsets in enumerate(enc["offset_mapping"]):
            # First token of each word inside this window. A token covering
            # several words (e.g. "62,") is shared by them.
            first_tok: dict[int, int] = {}
            word_i = 0
            for t, (ts, te) in enumerate(offsets):
                if ts == te:  # special token
                    continue
                while word_i < len(words) and words[word_i][1] <= ts:
                    word_i += 1
                j = word_i
                while j < len(words) and words[j][0] < te:
                    first_tok.setdefault(j, t)
                    j += 1
            if not first_tok:
                continue

            # Words cut by the window edges are only partly visible; drop them
            # (they appear whole in the neighbouring overlapping window).
            lo, hi = min(first_tok), max(first_tok)
            win_end = max(te for ts, te in offsets if ts != te)
            if words[lo][0] < offsets[first_tok[lo]][0]:
                lo += 1
            if words[hi][1] > win_end:
                hi -= 1
            if hi < lo:
                continue
            win_words = range(lo, hi + 1)

            spans = [
                (s - lo, e - lo, c) for s, e, c in doc_spans
                if s >= lo and e <= hi
            ]
            yield {
                "input_ids": torch.tensor(enc["input_ids"][w]),
                "attention_mask": torch.tensor(enc["attention_mask"][w]),
                "word_pos": torch.tensor([first_tok[i] for i in win_words]),
                "word_chars": torch.tensor([words[i] for i in win_words]),
                "spans": torch.tensor(spans, dtype=torch.long).reshape(-1, 3),
            }


class SpanCollator:
    """Pads a batch and builds ``SpanNER`` targets.

    ``targets[b, i * K + k, c] = 1`` iff words i..i+k of item b are an entity
    of class c, matching the ``(i, k)`` enumeration in ``SpanNER.forward``.
    """

    def __init__(self, pad_token_id: int, max_width: int = 12, num_labels: int = len(LABELS)):
        self.pad_token_id = pad_token_id
        self.K = max_width
        self.C = num_labels

    def __call__(self, batch):
        B = len(batch)
        T = max(len(x["input_ids"]) for x in batch)
        N = max(len(x["word_pos"]) for x in batch)

        input_ids = torch.full((B, T), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(B, T, dtype=torch.long)
        word_pos = torch.zeros(B, N, dtype=torch.long)
        word_mask = torch.zeros(B, N, dtype=torch.bool)
        targets = torch.zeros(B, N * self.K, self.C)

        for b, x in enumerate(batch):
            t, n = len(x["input_ids"]), len(x["word_pos"])
            input_ids[b, :t] = x["input_ids"]
            attention_mask[b, :t] = x["attention_mask"]
            word_pos[b, :n] = x["word_pos"]
            word_mask[b, :n] = True
            for s, e, c in x["spans"].tolist():
                targets[b, s * self.K + (e - s), c] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_pos": word_pos,
            "word_mask": word_mask,
            "targets": targets,
        }


def get_dataloader(dataset_paths: list[str], tokenizer, batch_size: int = 8, max_width: int = 12,
                   split: str = "train", shuffle: bool = True):
    data = pd.concat([pd.read_parquet(path) for path in dataset_paths])
    # Only the train split: the test split (grouped by patient) is for evaluation.
    data = data[(data["source"] == "synthetic") & (data["split"] == split)]
    dataset = Dataset(data, tokenizer, max_width=max_width)
    collator = SpanCollator(tokenizer.pad_token_id, max_width)
    # Shuffle so a batch isn't several overlapping windows of the same note.
    return tdata.DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=shuffle)


if __name__ == "__main__":
    dataset_path = "data/synthetic_notes.parquet"
    tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
    data = pd.read_parquet(dataset_path)
    data = data[data["source"] == "synthetic"]
    dataset = Dataset(data, tokenizer)
    print(f"{len(data)} notes -> {len(dataset)} windows, {dataset.too_wide} entities too wide")

    # Sanity check: decode the target spans of the first window back to text.
    item, text = dataset[0], data.iloc[0]["text"]
    for s, e, c in item["spans"].tolist():
        a, z = item["word_chars"][s][0].item(), item["word_chars"][e][1].item()
        print(f"{LABELS[c]:10s} {text[a:z]!r}")

    collator = SpanCollator(tokenizer.pad_token_id, max_width=dataset.max_width)
    loader = tdata.DataLoader(dataset, batch_size=8, collate_fn=collator)
    batch = next(iter(loader))
    print({k: tuple(v.shape) for k, v in batch.items()})
