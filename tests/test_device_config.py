"""
tests/test_device_config.py
============================
Unit tests for utils/device_config.py — hardware profiling utility.

Tests verify:
  - All 11 expected profile keys are present
  - Numeric values are in sensible, safe ranges
  - use_amp / pin_memory consistency with has_cuda
  - Atomic save helpers write files correctly

These tests run entirely in-process with no network, GPU, or dataset access.
"""

import json
import os
import tempfile

import pytest
import torch

from utils.device_config import (
    get_system_execution_profile,
    atomic_torch_save,
    atomic_json_save,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def profile():
    """Call the profiler once and share the result across tests in this module."""
    return get_system_execution_profile()


# ── Key presence ──────────────────────────────────────────────────────────────

EXPECTED_KEYS = {
    "device",
    "gpu_name",
    "has_cuda",
    "vram_gb",
    "total_ram_gb",
    "available_ram_gb",
    "batch_size",
    "num_workers",
    "use_amp",
    "pin_memory",
    "persistent_workers",
}

def test_profile_keys(profile):
    """All 11 expected keys must be present in the returned profile dict."""
    missing = EXPECTED_KEYS - set(profile.keys())
    assert not missing, f"Profile is missing keys: {missing}"


# ── Type correctness ──────────────────────────────────────────────────────────

def test_device_is_torch_device(profile):
    """profile['device'] must be a torch.device instance."""
    assert isinstance(profile["device"], torch.device), (
        f"Expected torch.device, got {type(profile['device'])}"
    )

def test_device_type_valid(profile):
    """Device type must be 'cuda' or 'cpu'."""
    assert profile["device"].type in ("cuda", "cpu"), (
        f"Unexpected device type: {profile['device'].type}"
    )

def test_gpu_name_is_string(profile):
    """gpu_name must be a non-empty string."""
    assert isinstance(profile["gpu_name"], str) and len(profile["gpu_name"]) > 0


# ── Numeric range checks ──────────────────────────────────────────────────────

def test_batch_size_positive(profile):
    """batch_size must be at least 1."""
    assert profile["batch_size"] >= 1, (
        f"batch_size must be ≥ 1, got {profile['batch_size']}"
    )

def test_batch_size_reasonable_upper_bound(profile):
    """batch_size should not exceed 512 (would always OOM on any current hardware)."""
    assert profile["batch_size"] <= 512, (
        f"batch_size suspiciously large: {profile['batch_size']}"
    )

def test_num_workers_non_negative(profile):
    """num_workers must be ≥ 0."""
    assert profile["num_workers"] >= 0, (
        f"num_workers must be ≥ 0, got {profile['num_workers']}"
    )

def test_num_workers_reasonable_upper_bound(profile):
    """num_workers should not exceed CPU count (would thrash the OS scheduler)."""
    import os
    cpu_count = os.cpu_count() or 4
    assert profile["num_workers"] <= cpu_count, (
        f"num_workers ({profile['num_workers']}) > cpu_count ({cpu_count})"
    )

def test_ram_values_positive(profile):
    """RAM values must be > 0."""
    assert profile["total_ram_gb"] > 0
    assert profile["available_ram_gb"] >= 0

def test_available_ram_leq_total(profile):
    """Available RAM cannot exceed total RAM."""
    assert profile["available_ram_gb"] <= profile["total_ram_gb"] + 0.1  # tiny float tolerance

def test_vram_non_negative(profile):
    """vram_gb must be ≥ 0 (0.0 on CPU-only systems)."""
    assert profile["vram_gb"] >= 0.0


# ── Boolean consistency checks ────────────────────────────────────────────────

def test_amp_requires_cuda(profile):
    """use_amp=True is only valid when has_cuda=True."""
    if profile["use_amp"]:
        assert profile["has_cuda"], (
            "use_amp=True but has_cuda=False — AMP is not supported on CPU"
        )

def test_pin_memory_requires_cuda(profile):
    """pin_memory=True is only valid when has_cuda=True."""
    if profile["pin_memory"]:
        assert profile["has_cuda"], (
            "pin_memory=True but has_cuda=False — pin_memory only helps CUDA DMA"
        )

def test_persistent_workers_iff_workers_positive(profile):
    """persistent_workers should be True iff num_workers > 0."""
    expected = profile["num_workers"] > 0
    assert profile["persistent_workers"] == expected, (
        f"persistent_workers={profile['persistent_workers']} but "
        f"num_workers={profile['num_workers']}"
    )

def test_no_cuda_means_no_amp(profile):
    """If has_cuda is False, use_amp must also be False."""
    if not profile["has_cuda"]:
        assert not profile["use_amp"], (
            "use_amp should be False on CPU-only systems"
        )


# ── Atomic save helpers ───────────────────────────────────────────────────────

def test_atomic_torch_save(tmp_path):
    """atomic_torch_save must write a valid state-dict without leaving a .tmp file."""
    target = str(tmp_path / "test_model.pth")
    state = {"weight": torch.tensor([1.0, 2.0, 3.0])}

    atomic_torch_save(state, target)

    assert os.path.exists(target), "Target file was not created"
    assert not os.path.exists(target + ".tmp"), ".tmp file was not cleaned up"

    loaded = torch.load(target, map_location="cpu", weights_only=True)
    assert torch.allclose(loaded["weight"], state["weight"])


def test_atomic_json_save(tmp_path):
    """atomic_json_save must write valid JSON without leaving a .tmp file."""
    target = str(tmp_path / "metrics.json")
    data = [{"epoch": 1, "val_f1": 0.92}, {"epoch": 2, "val_f1": 0.95}]

    atomic_json_save(data, target)

    assert os.path.exists(target), "Target file was not created"
    assert not os.path.exists(target + ".tmp"), ".tmp file was not cleaned up"

    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_atomic_json_save_overwrite(tmp_path):
    """atomic_json_save must overwrite an existing file atomically."""
    target = str(tmp_path / "overwrite_test.json")
    atomic_json_save({"v": 1}, target)
    atomic_json_save({"v": 2}, target)

    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["v"] == 2
