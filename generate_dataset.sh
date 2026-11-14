#!/usr/bin/env bash
# Generate the synthetic anonymization dataset with Qwen3-8B on an RTX 5090.
#
# The 8B model in bf16 (~16 GB weights + KV cache) fits in 32 GB VRAM.
# Override any knob via env vars, e.g.:
#   MODEL=Qwen/Qwen3-VL-4B-Instruct N_PATIENTS=10 ./generate_dataset.sh --max-tokens 1024
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false

python3 -m medical_imaging.anonymizer.dataset \
    --model "${MODEL:-Qwen/Qwen3-8B}" \
    --n-patients "${N_PATIENTS:-250}" \
    --docs-per-patient "${DOCS_PER_PATIENT:-2}" \
    --output "${OUTPUT:-data/synthetic_notes.parquet}" \
    --real-notes "${REAL_NOTES:-data/Export.parquet}" \
    --gpu-memory-utilization "${GPU_MEM:-0.85}" \
    "$@"
