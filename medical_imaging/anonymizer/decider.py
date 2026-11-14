import json
import pandas as pd
import torch
from transformers import AutoModel
from transformers.image_utils import load_image

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
data = pd.read_parquet("data/Export.parquet")
dtype = torch.float32 if device == "cpu" else torch.bfloat16
model = AutoModel.from_pretrained("LiquidAI/d1-3B", trust_remote_code=True, dtype=dtype).to(device)

# Text: several named questions over one state, answered in one pass
questions = {
    "is_anonymized": {
        "type": "noul",
        "instructions": "Nachádza sa v dokumente neanonymizovaný údaj ako napríklad meno, priezvisko, adresa, telefónne číslo, email, atď.?",
    },
    "what_type": {
        "type": "choice",
        "instructions": "Aky typ údajov sa nachadza v dokumente?",
        "criteria": {
            "meno": "Meno, priezvisko",
            "adresa": "Adresa",
            "telefon": "Telefónne číslo",
            "email": "Email",
            "none": "Žiadne osobné údaje",
        },
    },
}
print(model.system_one("Si odborný lekár, ktorý pracuje na zdravotníckej dokumentácii. Odpovedz na otázky.", questions))


# Batch: many requests, packed together with no padding

text_column = data["text"]

results = model.system_one_batch([(t, questions) for t in text_column])

for result, text in zip(results, text_column):
    if result["answers"]["is_anonymized"]["noul"] < 0.5:
        continue
    print(json.dumps(result, indent=2))
    print(text)
    print("-" * 80)