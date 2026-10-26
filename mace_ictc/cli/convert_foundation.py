"""Convert a trusted nonlinear MACE foundation model into an Ascend checkpoint.

Conversion runs on the scheduled compute resource in FP64. All original heads
are retained. The stored architecture adapter supports frozen force inference;
its effective matrices and embedding tables do not provide full fine-tuning.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def convert_file(source, output, *, head="omat_pbe", dtype=torch.float32):
    from mace_ictc.interfaces.foundation_hp import FoundationHP, load_foundation_checkpoint
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Checkpoint dtype must be float32 or float64")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        native = load_foundation_checkpoint(source).double().eval()
        model = FoundationHP(native, head=head).to(dtype=dtype).eval().requires_grad_(False)
        archive = dict(format="mace-ictc-foundation-ascend", version=1,
            source_mace_model=str(source), head=head, heads=model.heads,
            dtype=str(dtype).replace("torch.", ""), model=model,
            max_radius=model.max_radius, frozen_inference=True)
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(archive, path)
    finally:
        torch.set_default_dtype(previous)
    return dict(source=str(source), output=str(output), head=head, heads=model.heads,
        cutoff=model.max_radius, interactions=model.num_interaction,
        convolution_channels=[layer.channels for layer in model.convolutions],
        product_channels=[layer.channels for layer in model.products],
        dtype=archive["dtype"], frozen_inference=True, numerical_qualification_required=True)


def load_model(path, *, device="cpu", head=None):
    from mace_ictc.interfaces.foundation_hp import FoundationHP, load_foundation_checkpoint
    archive = load_foundation_checkpoint(path)
    if not isinstance(archive, dict) or archive.get("format") != "mace-ictc-foundation-ascend":
        raise ValueError("Expected a converted Ascend foundation checkpoint")
    model = archive.get("model")
    if archive.get("version") != 1 or not isinstance(model, FoundationHP):
        raise ValueError("Unsupported foundation archive version or model")
    if str(device).startswith("npu"):
        import torch_npu
        if archive.get("dtype") != "float32" or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("NPU inference requires an FP32 archive; convert explicitly with --dtype float32")
    if head is not None:model.select_head(head)
    return model.to(device=device).eval().requires_grad_(False)


def main(argv=None):
    from contextlib import redirect_stdout
    import json
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-model", required=True)
    parser.add_argument("--head", default="omat_pbe")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with redirect_stdout(sys.stderr):
        metadata = convert_file(args.mace_model, args.out, head=args.head,
                                dtype=getattr(torch, args.dtype))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":main()
