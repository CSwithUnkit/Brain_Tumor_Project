"""
utils/device_config.py
======================
Dynamic Hardware & System Profiling Utility for the MRI Brain Tumor Project.

Inspects the host machine at runtime and returns a safe, optimal execution
profile. Prevents OOM crashes on constrained RAM/VRAM systems and leverages
GPU Tensor Cores (AMP/fp16) when available.

Cross-platform: Windows, Linux, macOS, Google Colab.

Usage
-----
    from utils.device_config import get_system_execution_profile, print_profile

    profile = get_system_execution_profile()
    print_profile(profile)

    # Direct CLI invocation
    python -m utils.device_config
"""

import os
import sys

import psutil
import torch


def get_system_execution_profile() -> dict:
    """
    Dynamically inspect system resources and return an optimal, safe execution
    profile.

    Prevents Out-Of-Memory (OOM) crashes on constrained RAM systems and
    leverages GPU if present.

    Returns
    -------
    dict with keys:
        device              : torch.device  – 'cuda' or 'cpu'
        gpu_name            : str           – GPU model name or "CPU"
        has_cuda            : bool          – True when a CUDA GPU is available
        vram_gb             : float         – GPU total VRAM in GB (0.0 on CPU)
        total_ram_gb        : float         – System total RAM in GB
        available_ram_gb    : float         – System available RAM in GB
        batch_size          : int           – Safe default batch size
        num_workers         : int           – Safe DataLoader worker count
        use_amp             : bool          – Use torch.amp.autocast (fp16)
        pin_memory          : bool          – DataLoader pin_memory flag
        persistent_workers  : bool          – DataLoader persistent_workers flag
    """
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")

    # ── System RAM ────────────────────────────────────────────────────────────
    vm = psutil.virtual_memory()
    total_ram_gb     = vm.total    / (1024 ** 3)
    available_ram_gb = vm.available / (1024 ** 3)
    cpu_count        = os.cpu_count() or 4

    # ── GPU VRAM inspection ───────────────────────────────────────────────────
    vram_gb  = 0.0
    gpu_name = "CPU"
    if has_cuda:
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_name  = gpu_props.name
        vram_gb   = gpu_props.total_memory / (1024 ** 3)

    # ── Dynamic Safe Batch Size ───────────────────────────────────────────────
    # Respects VRAM tiers on GPU; falls back to RAM-aware sizes on CPU.
    if has_cuda:
        use_amp = True  # Automatic Mixed Precision (fp16) for Tensor Cores
        if vram_gb >= 10.0:
            batch_size = 32
        elif vram_gb >= 6.0:
            batch_size = 16
        else:
            batch_size = 8
    else:
        use_amp    = False
        batch_size = 4 if total_ram_gb <= 8.0 else 8

    # ── Safe DataLoader Worker Allocation ─────────────────────────────────────
    # CRITICAL: On Windows with ≤ 8 GB RAM, multiprocessing worker spawning
    # exhausts system memory (each worker forks a ~300 MB Python process).
    # On SATA SSDs or virtual sync drives (Google Drive desktop), excess workers
    # cause I/O lock contention and can deadlock the DataLoader.
    if os.name == "nt":  # Windows environment
        # Force num_workers = 0 on Windows to prevent PyTorch shared memory
        # page file allocation crashes (Error 1455: ERROR_COMMITMENT_LIMIT).
        num_workers = 0
    else:
        # Linux / macOS / Google Colab
        num_workers = 2 if total_ram_gb <= 8.5 else min(4, cpu_count)

    pin_memory         = has_cuda          # Pinned memory only helps CUDA DMA
    persistent_workers = num_workers > 0   # Keep workers alive between epochs

    return {
        "device"             : device,
        "gpu_name"           : gpu_name,
        "has_cuda"           : has_cuda,
        "vram_gb"            : round(vram_gb, 2),
        "total_ram_gb"       : round(total_ram_gb, 2),
        "available_ram_gb"   : round(available_ram_gb, 2),
        "batch_size"         : batch_size,
        "num_workers"        : num_workers,
        "use_amp"            : use_amp,
        "pin_memory"         : pin_memory,
        "persistent_workers" : persistent_workers,
    }


def print_profile(profile: dict) -> None:
    """Pretty-print the execution profile to stdout."""
    sep = "─" * 60
    print(sep)
    print("  🖥️  MRI Project — Hardware Execution Profile")
    print(sep)
    print(f"  Device          : {profile['device']} ({profile['gpu_name']})")
    if profile["has_cuda"]:
        print(f"  VRAM            : {profile['vram_gb']:.2f} GB")
    print(f"  Total RAM       : {profile['total_ram_gb']:.2f} GB  "
          f"(available: {profile['available_ram_gb']:.2f} GB)")
    print(f"  Batch Size      : {profile['batch_size']}")
    print(f"  DataLoader Workers: {profile['num_workers']}")
    print(f"  AMP (fp16)      : {profile['use_amp']}")
    print(f"  Pin Memory      : {profile['pin_memory']}")
    print(f"  Persistent Workers: {profile['persistent_workers']}")
    print(sep)


# ── Atomic file save helpers ──────────────────────────────────────────────────

def atomic_torch_save(obj, path: str) -> None:
    """
    Save a PyTorch object atomically to prevent write-corruption when running
    inside Google Drive synced folders or network file systems.

    Writes to a sibling `.tmp` file first, then uses os.replace() (atomic on
    POSIX; best-effort on Windows NTFS) to swap it into place.
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def atomic_json_save(data, path: str, indent: int = 4) -> None:
    """
    Save a JSON-serialisable object atomically to prevent write-corruption.
    """
    import json
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp_path, path)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    profile = get_system_execution_profile()
    print_profile(profile)
    # Convenience one-liner (matches notebook print requirement)
    print(
        f"\n🖥️ System Profile: {profile['gpu_name']} "
        f"({profile['vram_gb']} GB VRAM) | "
        f"RAM: {profile['total_ram_gb']} GB | "
        f"Workers: {profile['num_workers']}"
    )
