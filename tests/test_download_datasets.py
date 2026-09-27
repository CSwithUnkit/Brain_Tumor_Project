"""Regression tests for data/download_datasets.py.

All network interaction is hermetic: a local threaded HTTP server serves test
files, so these tests never touch Zenodo/Mendeley/the internet.

Covers the real failures seen in the field (2026-09-27/28):
- dropped mid-transfer connections (curl exit 18) -> resume + retry
- servers that ignore the Range header -> restart-from-scratch fallback
- the old notebook's fake-success bug -> content verification gates everything
"""

from __future__ import annotations

import functools
import os
import threading
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from data.download_datasets import (
    download_with_resume,
    ensure_brisc,
    ensure_pmram,
    extract_zip,
    verify_brisc,
    verify_pmram,
)


@pytest.fixture()
def http_server(tmp_path):
    """Serve tmp_path/'srv' over HTTP on a random local port."""
    srv = tmp_path / "srv"
    srv.mkdir()
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(srv))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, srv
    server.shutdown()


@pytest.fixture()
def payload(http_server):
    """A ~300 KiB pseudo-random file served over HTTP."""
    server, srv = http_server
    data = os.urandom(300 * 1024)
    (srv / "file.bin").write_bytes(data)
    url = f"http://127.0.0.1:{server.server_port}/file.bin"
    return url, data


def test_download_fresh(payload, tmp_path):
    url, data = payload
    dest = tmp_path / "out.bin"
    download_with_resume(url, dest, expected_size=len(data), max_retries=2)
    assert dest.read_bytes() == data


def test_download_resumes_partial(payload, tmp_path):
    """A half-written file must resume, not restart or corrupt."""
    url, data = payload
    dest = tmp_path / "out.bin"
    dest.write_bytes(data[: len(data) // 2])  # simulate dropped connection
    download_with_resume(url, dest, expected_size=len(data), max_retries=2)
    assert dest.read_bytes() == data


def test_download_restarts_when_server_ignores_range(tmp_path):
    """Servers that answer 200 to a Range request must trigger a clean restart."""

    class NoRangeHandler(SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if "Range" in self.headers:
                del self.headers["Range"]
            return super().do_GET()

    srv = tmp_path / "srv"
    srv.mkdir()
    data = os.urandom(200 * 1024)
    (srv / "file.bin").write_bytes(data)
    handler = functools.partial(NoRangeHandler, directory=str(srv))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/file.bin"
        dest = tmp_path / "out.bin"
        dest.write_bytes(b"stale-partial-bytes")
        download_with_resume(url, dest, expected_size=len(data), max_retries=2)
        assert dest.read_bytes() == data
    finally:
        server.shutdown()


def test_download_failure_raises_loudly(http_server, tmp_path):
    server, _srv = http_server
    url = f"http://127.0.0.1:{server.server_port}/does-not-exist.bin"
    with pytest.raises(RuntimeError, match="Could not download"):
        download_with_resume(url, tmp_path / "out.bin", max_retries=2)


def _make_brisc_tree(root: Path, n_cls: int = 3, n_seg: int = 2) -> None:
    base = root / "brisc2025"
    for split in ("train",):
        for cls in ("glioma", "meningioma", "pituitary", "no_tumor"):
            d = base / "classification_task" / split / cls
            d.mkdir(parents=True)
            for i in range(n_cls):
                (d / f"img_{i}.jpg").write_bytes(b"fake-jpg")
        img_d = base / "segmentation_task" / split / "images"
        msk_d = base / "segmentation_task" / split / "masks"
        img_d.mkdir(parents=True)
        msk_d.mkdir(parents=True)
        for i in range(n_seg):
            (img_d / f"s_{i}.jpg").write_bytes(b"fake-jpg")
            (msk_d / f"s_{i}.png").write_bytes(b"fake-png")


def test_verify_brisc_counts(tmp_path):
    _make_brisc_tree(tmp_path)
    stats = verify_brisc(tmp_path, min_classification=12, min_segmentation=2)
    assert stats == {"classification": 12, "segmentation_pairs": 2}
    with pytest.raises(RuntimeError, match="verification failed"):
        verify_brisc(tmp_path, min_classification=10_000)


def test_verify_brisc_rejects_missing_tree(tmp_path):
    with pytest.raises(RuntimeError, match="not found after extraction"):
        verify_brisc(tmp_path)


def _make_pmram_tree(root: Path) -> None:
    raw = root / "Raw"
    for folder, n in (
        ("512Glioma", 4),
        ("512Meningioma", 3),
        ("512Pituitary", 3),
        ("512Normal", 5),
    ):
        d = raw / folder
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img_{i}.jpg").write_bytes(b"fake-jpg")


def test_verify_pmram_counts(tmp_path):
    _make_pmram_tree(tmp_path)
    stats = verify_pmram(tmp_path, min_original=10)
    assert stats["original"] == 15
    with pytest.raises(RuntimeError, match="verification failed"):
        verify_pmram(tmp_path, min_original=10_000)


def test_verify_pmram_ignores_augmented(tmp_path):
    _make_pmram_tree(tmp_path)
    aug = tmp_path / "Raw" / "Augmented"
    aug.mkdir()
    (aug / "x.jpg").write_bytes(b"fake")
    stats = verify_pmram(tmp_path, min_original=10)
    assert stats["original"] == 15  # augmented folder excluded


def test_ensure_skips_download_when_verified(tmp_path, monkeypatch):
    """Idempotent skip: with verified content present, no network is attempted."""

    def _boom(*a, **k):
        raise AssertionError("network must not be used when content is verified")

    monkeypatch.setattr("data.download_datasets.download_with_resume", _boom)
    monkeypatch.setattr(
        "data.download_datasets.verify_brisc", lambda root, **k: {"classification": 1}
    )
    monkeypatch.setattr("data.download_datasets.verify_pmram", lambda root, **k: {"original": 1})
    assert ensure_brisc(tmp_path) == {"classification": 1}
    assert ensure_pmram(tmp_path) == {"original": 1}


def test_extract_zip_roundtrip(tmp_path):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "f.txt").write_text("hello")
    archive = tmp_path / "t.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(src / "a" / "f.txt", "a/f.txt")
    out = tmp_path / "out"
    extract_zip(archive, out)
    assert (out / "a" / "f.txt").read_text() == "hello"
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"this is not a zip file")
    with pytest.raises(RuntimeError, match="Cannot extract"):
        extract_zip(corrupt, tmp_path / "out2")
