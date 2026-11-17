from collections import Counter
import dataclasses

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

from medical_imaging.anonymizer.dataset import LABELS, Dataset, SpanCollator, get_dataloader
from medical_imaging.anonymizer.detector import get_gemma2_spanner


LABEL_STYLES = {
    "NAME": "bold white on red",
    "DOCTOR": "bold white on magenta",
    "AGE": "black on yellow",
    "DOB": "black on bright_yellow",
    "ID": "black on cyan",
    "ADDRESS": "black on green",
    "PHONE": "white on blue",
    "EMAIL": "black on bright_blue",
    "HOSPITAL": "black on bright_magenta",
}


@dataclasses.dataclass
class TrainingConfig:
    max_width: int = 18
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 10
    save_path: str = "models/gemma2_spanner.pth"


def focal_loss(logits, targets, valid, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = a_t * (1 - p_t) ** gamma * ce
    loss = loss * valid[..., None]
    return loss.sum() / targets.sum().clamp(min=1.0) 
    
    
@torch.no_grad()
def decode(logits, valid, K, thr=0.3):
    """Greedy 1D NMS: highest-scoring spans first, reject any overlap."""
    probs = logits.sigmoid() * valid[..., None]
    out = []
    for pb in probs:
        score, label = pb.max(-1)                                           # best label per span
        idx = (score > thr).nonzero().squeeze(-1)
        idx = idx[score[idx].argsort(descending=True)]
        taken, spans = set(), []
        for n in idx.tolist():
            s, e = n // K, n // K + n % K
            if taken.isdisjoint(range(s, e + 1)):
                taken.update(range(s, e + 1))
                spans.append((s, e, LABELS[label[n]], score[n].item()))
        out.append(sorted(spans))
    return out                     # normalize by #positives


def span_f1(preds, data):
    """Exact-match (start, end, label) precision/recall/F1 over character spans.

    Gold spans are whitespace-trimmed, matching the word-aligned predictions.
    """
    tp = n_pred = n_gold = 0
    for spans, ents in zip(preds, data["entities"]):
        gold = set()
        for en in ents:
            t = en["text"]
            gold.add((en["start"] + len(t) - len(t.lstrip()),
                      en["end"] - len(t) + len(t.rstrip()), en["label"]))
        pred = {(s, e, label) for s, e, label, _ in spans}
        tp += len(gold & pred)
        n_pred += len(pred)
        n_gold += len(gold)
    p = tp / max(n_pred, 1)
    r = tp / max(n_gold, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


class SpanNERTrainer:
    def __init__(self, config: TrainingConfig):
        self.model = get_gemma2_spanner(max_width=config.max_width)
        self.model.to("cuda")
        self.dtype = torch.bfloat16
        # Only the frozen encoder runs in bf16. Trainable weights must stay fp32:
        # in bf16 an AdamW step of ~lr=1e-4 is below the rounding step of most
        # weights (the -4.6 prior bias can't move at all), so the head never learns.
        self.model.enc.to(self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
        self.data_paths = ["data/synthetic_notes.parquet", "data/templated_notes.parquet"]
        self.dataset = get_dataloader(
            self.data_paths, 
            self.tokenizer, 
            max_width=config.max_width, 
            batch_size=config.batch_size
        )
        self.console = Console()
        enc_params = [p for p in self.model.enc.parameters() if p.requires_grad]
        enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in self.model.parameters()
                       if p.requires_grad and id(p) not in enc_ids]
        groups = [{"params": head_params, "lr": 1e-4},
                  {"params": enc_params, "lr": 1e-5}]
        self.optimizer = optim.AdamW([g for g in groups if g["params"]],
                                     weight_decay=0.01)

    def to_device(self, tensor_dict):
        return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in tensor_dict.items()}

    def forward(self, batch):
        with torch.autocast("cuda", dtype=self.dtype):
            logits, valid = self.model(batch["input_ids"], batch["attention_mask"],
                                       batch["word_pos"], batch["word_mask"])
        return logits.float(), valid

    def train(self, epochs: int = 10):
        self.model.train()
        self.model.enc.eval()  # frozen: no dropout noise in the features
        for epoch in range(epochs):
            self.evaluate()
            bar = tqdm(self.dataset, desc=f"epoch {epoch}")
            for batch in bar:
                batch = self.to_device(batch)
                logits, valid = self.forward(batch)
                loss = focal_loss(logits, batch["targets"], valid)

                self.optimizer.zero_grad(set_to_none=True)  # grads accumulated forever before
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                bar.set_postfix({"loss": loss.item()})

    @torch.no_grad()
    def predict(self, data, thr=0.3, batch_size=8):
        """Per row of ``data``: sorted (char_start, char_end, label, score) entities."""
        self.model.eval()
        K = self.model.K
        ds = Dataset(data, self.tokenizer, max_width=K)
        collate = SpanCollator(self.tokenizer.pad_token_id, K)
        found = [[] for _ in range(len(ds.data))]
        for b0 in range(0, len(ds), batch_size):
            items = ds.items[b0:b0 + batch_size]
            batch = self.to_device(collate(items))
            logits, valid = self.forward(batch)
            for j, spans in enumerate(decode(logits, valid, K, thr)):
                chars = items[j]["word_chars"]
                for s, e, label, score in spans:
                    found[ds.doc_index[b0 + j]].append(
                        (chars[s, 0].item(), chars[e, 1].item(), label, score))
        self.model.train()
        self.model.enc.eval()

        # Overlapping windows can predict the same entity twice: keep the best, drop overlaps.
        out = []
        for spans in found:
            kept = []
            for s, e, label, score in sorted(spans, key=lambda x: -x[3]):
                if all(e <= ks or s >= ke for ks, ke, _, _ in kept):
                    kept.append((s, e, label, score))
            out.append(sorted(kept))
        return out

    def evaluate(self, thr=0.3, n_docs=None, show_text=True):
        """Run inference on held-out synthetic test notes and print entities with rich."""
        data = pd.read_parquet(self.data_paths[0])
        # data = data[(data["source"] == "synthetic") & (data["split"] == "test")]
        data = data[data["source"] == "real"].sample(n=5, random_state=42)
        data = data.reset_index(drop=True)
        preds = self.predict(data, thr)
        p, r, f1 = span_f1(preds, data)
        self.console.print(f"[bold]test spans ({len(data)} notes): P={p:.3f} R={r:.3f} F1={f1:.3f}")
        n = n_docs or 5
        data, preds = data.head(n), preds[:n]

        self.console.rule(f"[bold]Inference on {len(data)} test notes (thr={thr})")
        self.console.print(Text(" ").join(
            Text(f" {label} ", style=LABEL_STYLES.get(label, "reverse")) for label in LABELS))

        counts = Counter()
        for (_, row), spans in zip(data.iterrows(), preds):
            text = row["text"]
            if show_text:
                highlighted = Text(text)
                for s, e, label, _ in spans:
                    highlighted.stylize(LABEL_STYLES.get(label, "reverse"), s, e)
                self.console.print(Panel(highlighted, border_style="cyan",
                                         title=f"{row['doc_id']} - {len(spans)} entities"))
            if spans:
                table = Table(title=row["doc_id"] if not show_text else None, show_lines=False)
                table.add_column("label")
                table.add_column("text")
                table.add_column("score", justify="right")
                table.add_column("chars", justify="right", style="dim")
                for s, e, label, score in spans:
                    counts[label] += 1
                    table.add_row(Text(label, style=LABEL_STYLES.get(label, "reverse")),
                                  repr(text[s:e]), f"{score:.2f}", f"{s}-{e}")
                self.console.print(table)

        summary = Table(title="Predicted entities per label")
        summary.add_column("label")
        summary.add_column("count", justify="right")
        for label in LABELS:
            summary.add_row(Text(label, style=LABEL_STYLES.get(label, "reverse")), str(counts[label]))
        self.console.print(summary)
        return preds


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = SpanNERTrainer(config)
    trainer.train(config.epochs)
    trainer.evaluate()
