"""
deployment/export_model.py
===========================
Automated model export utility for the MRI Brain Tumor Project.

Packages trained PyTorch checkpoints into multiple deployment formats:

  • PyTorch state-dict (.pth)   — clean, optimizer-stripped checkpoint
  • TorchScript traced (.pt)    — self-contained for C++ / mobile / no-Python envs
  • ONNX (.onnx)                — dynamic batch axis, opset 17, ORT validation

Cross-platform: Windows, Linux, macOS, Google Colab.

Usage (CLI)
-----------
    python -m deployment.export_model \\
        --model classifier \\
        --checkpoint checkpoints/classification/best_efficientnet_exp2_enhanced.pth \\
        --output-dir deployment/exported/ \\
        --output-format onnx

    python -m deployment.export_model \\
        --model unet \\
        --checkpoint checkpoints/unet/best_unet_enhanced.pth \\
        --output-format all
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch

# ── Conditional soft imports (graceful degradation) ───────────────────────────
try:
    import onnx
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSIFIER_INPUT_SHAPE = (1, 3, 256, 256)   # EfficientNet-B2
UNET_INPUT_SHAPE       = (1, 3, 256, 256)   # U-Net
ONNX_OPSET_VERSION     = 17
ORT_MAX_DELTA          = 1e-4               # max abs diff for numerical validation


# ── Model factory ─────────────────────────────────────────────────────────────

def _build_classifier() -> torch.nn.Module:
    """Instantiate BrainTumorClassifier (EfficientNetB2, 4 classes)."""
    # Local import so the module is usable without being in the project root
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from classification.classifier_model import BrainTumorClassifier
    return BrainTumorClassifier(num_classes=4, pretrained=False)


def _build_unet() -> torch.nn.Module:
    """Instantiate U-Net (3→1 channels)."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from segmentation.unet_model import UNet
    return UNet(n_channels=3, n_classes=1)


def _load_model(model_type: str, checkpoint_path: str) -> tuple[torch.nn.Module, tuple]:
    """Load model from checkpoint and return (model_on_cpu, input_shape)."""
    if model_type == "classifier":
        model      = _build_classifier()
        input_shape = CLASSIFIER_INPUT_SHAPE
    elif model_type == "unet":
        model      = _build_unet()
        input_shape = UNET_INPUT_SHAPE
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose 'classifier' or 'unet'.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first, or pass --checkpoint with the correct path."
        )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Handle wrapped checkpoints (e.g. {'model_state_dict': ...})
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    logger.info(f"✓ Loaded {model_type} from {checkpoint_path}")
    return model, input_shape


# ── Export functions ──────────────────────────────────────────────────────────

def export_state_dict(model: torch.nn.Module, output_path: str) -> None:
    """
    Save a clean state-dict only checkpoint (optimizer states stripped).
    Smallest deployment artefact; can be reloaded with model.load_state_dict().
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp = output_path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, output_path)
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    logger.info(f"✓ State-dict saved → {output_path}  ({size_mb:.1f} MB)")


def export_torchscript(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
) -> None:
    """
    Export a TorchScript traced model (.pt).
    Compatible with C++ LibTorch, mobile, and environments without Python source.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dummy_input = torch.zeros(*input_shape)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
    tmp = output_path + ".tmp"
    traced.save(tmp)
    os.replace(tmp, output_path)
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    logger.info(f"✓ TorchScript saved → {output_path}  ({size_mb:.1f} MB)")


def export_onnx(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
    model_type: str,
) -> None:
    """
    Export model to ONNX format with dynamic batch axis and opset 17.
    If onnxruntime is installed, runs a numerical validation pass.
    """
    if not _ONNX_AVAILABLE:
        logger.error(
            "onnx package not installed. Install with:\n"
            "    pip install onnx onnxruntime\n"
            "Then re-run this command."
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dummy_input = torch.zeros(*input_shape)

    # Input/output names vary by model type
    if model_type == "classifier":
        input_names  = ["mri_image"]
        output_names = ["class_logits"]
    else:
        input_names  = ["mri_image"]
        output_names = ["tumor_mask_logits"]

    # Dynamic batch axis: allows arbitrary batch sizes at inference time
    dynamic_axes = {
        input_names[0]:  {0: "batch_size"},
        output_names[0]: {0: "batch_size"},
    }

    tmp_path = output_path + ".tmp"
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            tmp_path,
            opset_version=ONNX_OPSET_VERSION,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    # ── ONNX graph integrity check ─────────────────────────────────────────
    onnx_model = onnx.load(tmp_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✓ ONNX graph integrity check passed")

    os.replace(tmp_path, output_path)
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    logger.info(f"✓ ONNX model saved → {output_path}  ({size_mb:.1f} MB)")

    # ── Numerical validation via ORT ───────────────────────────────────────
    if _ORT_AVAILABLE:
        _validate_onnx_vs_pytorch(model, dummy_input, output_path, model_type)
    else:
        logger.warning(
            "onnxruntime not installed — skipping numerical validation.\n"
            "Install with: pip install onnxruntime"
        )


def _validate_onnx_vs_pytorch(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    model_type: str,
) -> None:
    """Run PyTorch and ORT on the same input and compare outputs numerically."""
    import numpy as np

    # PyTorch reference output
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    # ORT output
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {ort_input_name: dummy_input.numpy()})[0]

    max_delta = float(np.max(np.abs(pt_out - ort_out)))
    if max_delta <= ORT_MAX_DELTA:
        logger.info(
            f"✓ Numerical validation PASSED — max |PyTorch − ORT| = {max_delta:.2e} "
            f"(threshold {ORT_MAX_DELTA:.0e})"
        )
    else:
        logger.warning(
            f"⚠ Numerical validation WARNING — max |PyTorch − ORT| = {max_delta:.2e} "
            f"exceeds threshold {ORT_MAX_DELTA:.0e}. "
            "Inspect the ONNX graph for unsupported ops or precision loss."
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m deployment.export_model",
        description="Export trained MRI tumor detection models to PyTorch / TorchScript / ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Export classifier to ONNX
  python -m deployment.export_model \\
      --model classifier \\
      --checkpoint checkpoints/classification/best_efficientnet_exp2_enhanced.pth \\
      --output-format onnx

  # Export U-Net to all formats
  python -m deployment.export_model \\
      --model unet \\
      --checkpoint checkpoints/unet/best_unet_enhanced.pth \\
      --output-dir deployment/exported/ \\
      --output-format all
        """,
    )
    parser.add_argument(
        "--model", choices=["classifier", "unet"], required=True,
        help="Which model to export."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained .pth checkpoint file."
    )
    parser.add_argument(
        "--output-dir", type=str, default="deployment/exported",
        help="Directory to write exported files into. (default: deployment/exported/)"
    )
    parser.add_argument(
        "--output-format", choices=["pth", "torchscript", "onnx", "all"],
        default="all",
        help="Export format(s). 'all' exports pth + torchscript + onnx. (default: all)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, input_shape = _load_model(args.model, args.checkpoint)

    stem = f"{args.model}_exported"
    fmt  = args.output_format

    if fmt in ("pth", "all"):
        export_state_dict(
            model,
            os.path.join(args.output_dir, f"{stem}.pth"),
        )

    if fmt in ("torchscript", "all"):
        export_torchscript(
            model, input_shape,
            os.path.join(args.output_dir, f"{stem}.pt"),
        )

    if fmt in ("onnx", "all"):
        export_onnx(
            model, input_shape,
            os.path.join(args.output_dir, f"{stem}.onnx"),
            model_type=args.model,
        )

    logger.info(f"\n✅ Export complete. Files written to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
