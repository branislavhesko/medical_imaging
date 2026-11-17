from transformers import AutoModel
import torch
import torch.nn as nn

from medical_imaging.anonymizer.dataset import LABELS


class SpanNER(nn.Module):
    def __init__(self, encoder, hidden, num_labels=len(LABELS), max_width=12, dropout=0.1):
        super().__init__()
        self.enc, self.K = encoder, max_width
        mlp = lambda: nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.start, self.end = mlp(), mlp()
        self.width = nn.Embedding(max_width, hidden // 4)          # optional width prior
        self.out = nn.Sequential(nn.Linear(2 * hidden + hidden // 4, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, num_labels))
        nn.init.constant_(self.out[-1].bias, -4.6)                  # prior p≈0.01, as in RetinaNet
        # self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.enc.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, word_pos, word_mask):
        # word_pos: [B, N] index of first subword of each word; word_mask: [B, N] bool
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h.gather(1, word_pos[..., None].expand(-1, -1, h.size(-1)))   # [B, N, D]
        B, N, _ = h.shape
        K = self.K
        i = torch.arange(N, device=h.device).repeat_interleave(K)          # [N*K] starts
        k = torch.arange(K, device=h.device).repeat(N)                     # [N*K] widths-1
        j = i + k                                                           # ends
        n_words = word_mask.sum(1, keepdim=True)                            # [B, 1]
        valid = j[None] < n_words                                           # [B, N*K]
        jc = j.clamp(max=N - 1)
        s = torch.cat([self.start(h)[:, i], self.end(h)[:, jc],
                       self.width(k)[None].expand(B, -1, -1)], dim=-1)
        logits = self.out(s)                                                # [B, N*K, C]
        return logits, valid


def get_gemma2_spanner(
    model_path: str = "google/embeddinggemma-2",
    hidden: int = 768,
    num_labels: int = len(LABELS),
    max_width: int = 12,
    dropout: float = 0.1,
):
    enc = AutoModel.from_pretrained(model_path)   # text tower only
    return SpanNER(
        enc, 
        hidden=hidden, 
        num_labels=num_labels, 
        max_width=max_width, 
        dropout=dropout
    )


if __name__ == "__main__":
    model = get_gemma2_spanner()
    print(model)
