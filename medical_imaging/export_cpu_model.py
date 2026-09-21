"""Rebuild a device-portable ``.pt2`` from a checkpoint exported on CUDA.

``torch.export.save`` bakes the export device into the archive (weights,
example inputs and graph metadata), so a model exported on a GPU box cannot be
loaded by CPU-only torch. This pulls the raw weights out of such an archive,
loads them into a fresh timm model and re-exports on CPU with a dynamic batch
dimension::

    python -m medical_imaging.export_cpu_model models/model_best_cuda.pt2 models/model_best.pt2
"""

import argparse

from omegaconf import OmegaConf
import torch
import torch.export.pt2_archive._package as pkg
from torch.export.pt2_archive import PT2ArchiveReader

from medical_imaging.train_classifier import export_model


def load_state_dict_any_device(pt2_path: str) -> dict[str, torch.Tensor]:
    """Read the weights of a ``.pt2`` archive onto the CPU regardless of the
    device they were saved from."""
    original = pkg.deserialize_device
    pkg.deserialize_device = lambda _: torch.device("cpu")
    try:
        with PT2ArchiveReader(pt2_path) as reader:
            model_names = sorted(
                name.split("/")[-1].removesuffix(".json")
                for name in reader.get_file_names()
                if name.startswith(pkg.MODELS_DIR)
            )
            if len(model_names) != 1:
                raise ValueError(f"Expected one model in {pt2_path}, found {model_names}")
            state_dict = pkg._load_state_dict(reader, model_names[0])
    finally:
        pkg.deserialize_device = original
    if not isinstance(state_dict, dict):
        raise ValueError("Legacy pickled weights are not supported; re-export the model")
    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="existing .pt2 (any device)")
    parser.add_argument("target", help="where to write the CPU-exported .pt2")
    parser.add_argument("--config", default="models/model_best.yaml")
    args = parser.parse_args()

    from timm import create_model

    config = OmegaConf.load(args.config)
    model = create_model(config.model_name, pretrained=False, num_classes=config.num_classes)
    model.load_state_dict(load_state_dict_any_device(args.source))
    export_model(model, args.target)
    print(f"Wrote {args.target}")


if __name__ == "__main__":
    main()
