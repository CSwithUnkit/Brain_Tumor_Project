import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#0b0f14"
PANEL = "#111820"
BORDER = "#1f2f3f"
CYAN = "#00d2ff"
TEAL = "#00b5cc"
GREEN = "#10d97a"
RED = "#ef4455"
AMBER = "#f59e0b"
MUTED = "#5a7a90"
TEXT = "#cdd9e5"
WHITE = "#f0f6fc"


def _apply_dark_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": CYAN,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": BORDER,
            "grid.linewidth": 0.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "legend.facecolor": PANEL,
            "legend.edgecolor": BORDER,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": BG,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
        }
    )


class ReportConsolidator:
    """
    FR-043–047: Generates publication-quality medical AI evaluation reports.
    Produces dark-themed matplotlib figures and structured Markdown documents.
    """

    def __init__(self, project_root: str | None = None):
        # Resolve paths relative to the project root. Defaults to this file's
        # location so main() works regardless of cwd (e.g. Notebook 3 Cell 8),
        # but tests / callers can inject an isolated directory — previously
        # the paths were ALWAYS the real repo, so the test suite overwrote
        # the actual reports/ directory (fixed).
        _ROOT = (
            Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
        )
        self.project_root = str(_ROOT)
        self.results_dir = str(_ROOT / "results")
        self.reports_dir = str(_ROOT / "reports")
        self.figures_dir = str(_ROOT / "reports" / "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        _apply_dark_style()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _load(self, path: str) -> Any:
        if not os.path.exists(path):
            logger.warning(f"Not found: {path}")
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    def _best_epoch(self, history: list) -> dict | None:
        if not history:
            return None
        return max(history, key=lambda h: h.get("val_metrics", {}).get("macro_f1", 0))

    def _pct(self, v) -> str:
        if isinstance(v, (int, float)):
            return f"{v * 100:.2f}%"
        return "N/A"

    # ── Figure 1: Classifier Comparison Bar Chart ──────────────────────────────

    def _plot_classifier_comparison(self, summary: dict[str, dict]) -> str:
        """Side-by-side grouped bar chart for all 3 experiments × 4 metrics."""
        if not summary:
            return ""

        experiments = list(summary.keys())
        metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
        labels = ["Accuracy", "Macro F1", "Precision", "Recall"]
        colors = [CYAN, GREEN, AMBER, "#a78bfa"]

        x = np.arange(len(experiments))
        width = 0.18
        fig, ax = plt.subplots(figsize=(11, 5))

        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors, strict=True)):
            vals = [summary[e].get(metric, 0) * 100 for e in experiments]
            bars = ax.bar(
                x + i * width,
                vals,
                width,
                label=label,
                color=color,
                alpha=0.88,
                edgecolor=BG,
                linewidth=0.5,
            )
            for bar, val in zip(bars, vals, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=WHITE,
                    fontweight="bold",
                )

        ax.set_xlabel("Experiment", labelpad=8)
        ax.set_ylabel("Score (%)", labelpad=8)
        ax.set_title(
            "EfficientNetB2 — Classification Performance Across All Experiments",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([e.replace("_", " ").title() for e in experiments], fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", ncol=4, framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        save_path = os.path.join(self.figures_dir, "model_comparison.png")
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    # ── Figure 2: Training Convergence Curves ─────────────────────────────────

    def _plot_convergence(self, histories: dict[str, list]) -> str:
        """Multi-experiment F1 convergence curves on one chart."""
        colors_map = {
            "exp1_baseline": (CYAN, "-"),
            "exp2_enhanced": (GREEN, "--"),
            "exp3_seg_guided": (AMBER, "-."),
        }
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = False
        for exp, history in histories.items():
            if not history:
                continue
            epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
            f1s = [h.get("val_metrics", {}).get("macro_f1", 0) * 100 for h in history]
            color, ls = colors_map.get(exp, ("#ffffff", "-"))
            ax.plot(
                epochs,
                f1s,
                ls=ls,
                color=color,
                linewidth=2,
                label=exp.replace("_", " ").title(),
                marker="o",
                markersize=3,
                markerfacecolor=color,
                alpha=0.9,
            )
            # Annotate best
            best_f1 = max(f1s)
            best_ep = epochs[f1s.index(best_f1)]
            ax.annotate(f"  {best_f1:.1f}%", xy=(best_ep, best_f1), fontsize=7.5, color=color)
            plotted = True

        if not plotted:
            plt.close(fig)
            return ""

        ax.set_xlabel("Epoch", labelpad=8)
        ax.set_ylabel("Macro F1 (%)", labelpad=8)
        ax.set_title(
            "Validation Macro-F1 Convergence — All Experiments",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        ax.legend(loc="lower right", framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        save_path = os.path.join(self.figures_dir, "convergence_curves.png")
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    # ── Figure 3: Generalization Gap ──────────────────────────────────────────

    def _plot_generalization(self, brisc_acc: float, pmram_acc: float) -> str:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cats = ["BRISC\n(Internal Test)", "PMRAM\n(External Validation)"]
        vals = [brisc_acc * 100, pmram_acc * 100]
        bar_c = [CYAN, AMBER]
        bars = ax.bar(cats, vals, color=bar_c, width=0.45, edgecolor=BG, linewidth=0.8, alpha=0.9)
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                color=WHITE,
                fontweight="bold",
            )

        gap = (brisc_acc - pmram_acc) * 100
        ax.annotate(
            "",
            xy=(1, pmram_acc * 100),
            xytext=(1, brisc_acc * 100),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=2),
        )
        ax.text(
            1.22,
            (brisc_acc + pmram_acc) * 50,
            f"Δ = {gap:.1f}%",
            color=RED,
            fontsize=9,
            fontweight="bold",
        )

        ax.set_ylim(0, 115)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(
            "Generalization Gap: Internal vs External Validation",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        save_path = os.path.join(self.figures_dir, "generalization_gap.png")
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    # ── Figure 4: U-Net Dice Convergence ──────────────────────────────────────

    def _plot_unet_convergence(self, history: list) -> str:
        if not history:
            return ""
        epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
        val_dice = [h.get("val_dice", 0) for h in history]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.fill_between(epochs, val_dice, alpha=0.12, color=TEAL)
        ax.plot(
            epochs,
            val_dice,
            color=TEAL,
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label="Val Dice",
        )
        best = max(val_dice)
        best_ep = epochs[val_dice.index(best)]
        ax.axhline(best, ls="--", color=CYAN, alpha=0.6, linewidth=1.2, label=f"Best = {best:.4f}")
        ax.scatter([best_ep], [best], color=CYAN, s=80, zorder=5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice Coefficient")
        ax.set_title(
            "U-Net Segmentation — Validation Dice Convergence",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        ax.legend(framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        save_path = os.path.join(self.figures_dir, "unet_dice_convergence.png")
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    # ── Markdown Generators ───────────────────────────────────────────────────

    def _horizontal_rule(self) -> str:
        return "\n---\n"

    def _metric_table(self, rows: list, headers: list) -> str:
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        head = "| " + " | ".join(headers) + " |"
        body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
        return f"{head}\n{sep}\n{body}"

    def generate_phase1_report(self):
        logger.info("Generating Phase I Report...")

        # Load data
        hist1 = self._load(f"{self.results_dir}/metrics_exp1_baseline.json") or []
        hist2 = self._load(f"{self.results_dir}/metrics_exp2_enhanced.json") or []
        unet = (
            self._load(f"{self.results_dir}/unet_metrics.json")
            or self._load(
                f"{self.project_root}/checkpoints/unet/unet_training_history_enhanced.json"
            )
            or []
        )

        b1 = self._best_epoch(hist1) or {}
        b2 = self._best_epoch(hist2) or {}
        m1 = b1.get("val_metrics", {})
        m2 = b2.get("val_metrics", {})

        # M4c: dataset counts from the actual ingested metadata, not hardcoded.
        meta = self._load(f"{self.project_root}/data/brisc/brisc_metadata.json") or {}
        n_cls = meta.get("classification_count")
        n_seg = meta.get("segmentation_count")
        dataset_line = (
            f"BRISC 2025 ({n_cls:,} classification + {n_seg:,} segmentation pairs)"
            if isinstance(n_cls, int) and isinstance(n_seg, int)
            else "BRISC 2025 (counts unavailable — run ingestion)"
        )

        # Generate figures
        summary = {}
        if m1:
            summary["Exp 1 Baseline"] = m1
        if m2:
            summary["Exp 2 Enhanced"] = m2
        fig_bar = self._plot_classifier_comparison(summary)
        fig_conv = self._plot_convergence({"exp1_baseline": hist1, "exp2_enhanced": hist2})
        fig_unet = self._plot_unet_convergence(unet)

        best_unet_dice_raw = max((h.get("val_dice", 0) for h in unet), default=None)
        best_unet_dice = (
            f"{best_unet_dice_raw:.4f}" if isinstance(best_unet_dice_raw, float) else "N/A"
        )
        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        rows = [
            [
                "Experiment 1 — Baseline (Raw)",
                self._pct(m1.get("accuracy")),
                self._pct(m1.get("macro_f1")),
                self._pct(m1.get("macro_precision")),
                self._pct(m1.get("macro_recall")),
                b1.get("epoch", "N/A"),
            ],
            [
                "Experiment 2 — Enhanced (WPT+LMMSE+CLAHE)",
                self._pct(m2.get("accuracy")),
                self._pct(m2.get("macro_f1")),
                self._pct(m2.get("macro_precision")),
                self._pct(m2.get("macro_recall")),
                b2.get("epoch", "N/A"),
            ],
        ]
        headers = ["Experiment", "Accuracy", "Macro F1", "Precision", "Recall", "Best Epoch"]

        report = f"""# PHASE I EVALUATION REPORT
## MRI Image Enhancing and Tumor Detection
### An Explainable Deep Learning Framework for Brain Tumor Segmentation and Classification

> **Institution:** ITS Engineering College, AKTU  
> **Generated:** {ts}  
> **Dataset:** {dataset_line}

{self._horizontal_rule()}

## Executive Summary

This Phase I report evaluates the baseline EfficientNetB2 classification pipeline and the U-Net
tumor segmentation network. Two training configurations are compared: raw MRI inputs (Exp 1)
versus images pre-processed with the WPT→LMMSE→CLAHE enhancement pipeline (Exp 2).

{self._horizontal_rule()}

## 1. U-Net Tumor Segmentation

| Parameter | Value |
|---|---|
| Architecture | U-Net (3-channel input, 1 binary output) |
| Loss Function | 0.4 × BCE (pos_weight=10) + 0.6 × Tversky-Focal (α=0.7, β=0.3, γ=0.75) |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Training Pairs | {len(unet)} epochs recorded |
| **Best Val Dice** | **{best_unet_dice}** |

{"![U-Net Convergence](figures/unet_dice_convergence.png)" if fig_unet else "*Checkpoint history not yet available.*"}

{self._horizontal_rule()}

## 2. Classification Experiments — Exp 1 vs Exp 2

### 2.1 Architecture & Training Protocol

| Component | Configuration |
|---|---|
| Backbone | EfficientNetB2 (ImageNet pretrained) |
| Head | GlobalAvgPool → Dropout(0.3) → Linear(4 classes) |
| Stage 1 (Epochs 1–5) | Backbone frozen, Head warmup with LinearLR |
| Stage 2 (Epochs 6–25) | Full fine-tuning with differential learning rates (backbone: 1e-5, head: 1e-4) |
| Loss | CrossEntropy (label_smoothing=0.1) |
| Augmentation | Affine + HorizontalFlip + ElasticTransform + GaussianBlur + BrightnessContrast |
| Gradient Clipping | max_norm = 1.0 |

### 2.2 Results Table

{self._metric_table(rows, headers)}

### 2.3 Performance Visualization

{"![Classifier Comparison](figures/model_comparison.png)" if fig_bar else "*Figures not yet available — run experiments first.*"}

{"![Convergence Curves](figures/convergence_curves.png)" if fig_conv else ""}

{self._horizontal_rule()}

## 3. Key Observations

- Enhancement (Exp 2) is expected to improve CNR and tumour boundary delineation
- WPT→LMMSE→CLAHE caching eliminates on-the-fly CPU processing during GPU training
- Class imbalance addressed via label smoothing + pos_weight=10 on the segmentation BCE term

{self._horizontal_rule()}

*Report auto-generated by `evaluation/consolidate_reports.py` — MRI Image Enhancing and Tumor Detection*
"""
        path = os.path.join(self.reports_dir, "PHASE_I_EVALUATION_REPORT.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Phase I report saved → {path}")

    def generate_phase2_report(self):
        logger.info("Generating Phase II Report...")

        hist3 = self._load(f"{self.results_dir}/metrics_exp3_seg_guided.json") or []
        hist1 = self._load(f"{self.results_dir}/metrics_exp1_baseline.json") or []
        hist2 = self._load(f"{self.results_dir}/metrics_exp2_enhanced.json") or []
        # M2: held-out TEST metrics (best checkpoint, never seen in train/val).
        # Absent before the 2026-09-27 fixes; their presence marks a post-fix run.
        t1 = self._load(f"{self.results_dir}/metrics_exp1_baseline_test.json") or {}
        t2 = self._load(f"{self.results_dir}/metrics_exp2_enhanced_test.json") or {}
        t3 = self._load(f"{self.results_dir}/metrics_exp3_seg_guided_test.json") or {}
        has_test = any([t1, t2, t3])
        xai = self._load(f"{self.results_dir}/gradcam_localization_summary.json") or {}
        pmram = self._load(f"{self.results_dir}/pmram_external_validation.json") or {}

        b1 = self._best_epoch(hist1) or {}
        b2 = self._best_epoch(hist2) or {}
        b3 = self._best_epoch(hist3) or {}
        m1 = b1.get("val_metrics", {})
        m2 = b2.get("val_metrics", {})
        m3 = b3.get("val_metrics", {})

        pmram_m = pmram.get("pmram_metrics", {})
        brisc_m = pmram.get("brisc_metrics", {})
        gen_gap = pmram.get("generalization_gap", {})

        brisc_acc = brisc_m.get("accuracy", m3.get("accuracy", None))
        pmram_acc = pmram_m.get("accuracy", None)

        # Figures
        fig_all = self._plot_classifier_comparison(
            {
                "Exp 1 Baseline": m1,
                "Exp 2 Enhanced": m2,
                "Exp 3 Seg-Guided": m3,
            }
        )
        fig_conv = self._plot_convergence(
            {
                "exp1_baseline": hist1,
                "exp2_enhanced": hist2,
                "exp3_seg_guided": hist3,
            }
        )
        fig_gap = ""
        if isinstance(brisc_acc, float) and isinstance(pmram_acc, float):
            fig_gap = self._plot_generalization(brisc_acc, pmram_acc)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        pmram_n = pmram.get("num_samples_evaluated", "N/A")

        # M2: provenance note reflects what the files actually contain — it
        # must not stay stale after a genuine post-fix re-run. Test-metric
        # files only exist for post-fix runs (the pre-fix code never wrote
        # held-out test metrics).
        if has_test:
            provenance_note = (
                "> **Metrics provenance:** validation columns show the "
                "best-validation-epoch metrics used for checkpoint selection; "
                "test columns show the HELD-OUT test set (15% stratified split, "
                "never used in training or validation) scored with the best "
                "checkpoint. Exp 3 applies identical segmentation-guided "
                "masking in train/val/test."
            )
        else:
            provenance_note = (
                "> **⚠️ Metrics provenance:** no held-out test metrics were found "
                "(results/metrics_*_test.json missing) — the numbers below are "
                "best-validation-epoch values from runs that pre-date the "
                "2026-09-27 code fixes, and Exp 3's come from runs where "
                "segmentation masking was applied in training but **not** in "
                "validation. Do not cite; re-run the full pipeline first."
            )

        exp3_label = (
            "Exp 3 — Segmentation-Guided" if has_test else "Exp 3 — Segmentation-Guided ⚠️ pre-fix"
        )
        all_rows = [
            [
                "Exp 1 — Baseline (Raw)",
                self._pct(m1.get("accuracy")),
                self._pct(m1.get("macro_f1")),
                self._pct(t1.get("accuracy")),
                self._pct(t1.get("macro_f1")),
                b1.get("epoch", "N/A"),
            ],
            [
                "Exp 2 — Enhanced (WPT+LMMSE+CLAHE)",
                self._pct(m2.get("accuracy")),
                self._pct(m2.get("macro_f1")),
                self._pct(t2.get("accuracy")),
                self._pct(t2.get("macro_f1")),
                b2.get("epoch", "N/A"),
            ],
            [
                exp3_label,
                self._pct(m3.get("accuracy")),
                self._pct(m3.get("macro_f1")),
                self._pct(t3.get("accuracy")),
                self._pct(t3.get("macro_f1")),
                b3.get("epoch", "N/A"),
            ],
        ]
        headers = [
            "Experiment",
            "Val Accuracy",
            "Val Macro F1",
            "Test Accuracy",
            "Test Macro F1",
            "Best Epoch",
        ]

        xai_rows = []
        if isinstance(xai, dict):
            for exp_name, thr_data in xai.items():
                if not isinstance(thr_data, dict):
                    continue

                def _thr_key(kv):
                    try:
                        return float(kv[0])
                    except (TypeError, ValueError):
                        return 0.0

                for t_str, m in sorted(thr_data.items(), key=_thr_key):
                    if not isinstance(m, dict):
                        continue
                    xai_rows.append(
                        [
                            exp_name,
                            t_str,
                            f"{m.get('mean_iou', 0):.4f} ± {m.get('std_iou', 0):.4f}",
                            f"{m.get('mean_dice', 0):.4f} ± {m.get('std_dice', 0):.4f}",
                            str(m.get("n_samples", "N/A")),
                        ]
                    )

        pmram_has_provenance = bool(pmram.get("checkpoint_sha256"))
        _pmram_ckpt = pmram.get("checkpoint_path", "unknown")
        _pmram_prep = pmram.get("preprocessing", "unknown")
        if pmram_has_provenance:
            pmram_caveats = (
                "> **Provenance:** the evaluated checkpoint and its SHA-256 are recorded in "
                "results/pmram_external_validation.json "
                f"(`{_pmram_ckpt}`), evaluated with `{_pmram_prep}` preprocessing — "
                "the input distribution matching that model's training. The gap in §3 is "
                "measured on that basis."
            )
        else:
            pmram_caveats = (
                '> **Caveats:** the "BRISC Internal" reference stored alongside the PMRAM results '
                "matches Exp 3's best validation epoch, **not** the Exp 1 baseline, and the "
                "validation script does not record which checkpoint was evaluated — treat the "
                "gap as indicative, not definitive. A gap of < 5% would indicate strong "
                "cross-domain robustness *once measured against a matched reference*."
            )
        if has_test:
            conclusion_1 = (
                "1. **Enhancement impact:** compare the Test Accuracy / Test Macro F1 "
                "columns above — Exp 2 (WPT→LMMSE→CLAHE) vs the Exp 1 raw baseline. "
                "A consistent test-set lead for Exp 2 would support the enhancement "
                "claim; judge only on the held-out test numbers, not validation."
            )
            conclusion_2 = (
                "2. **Segmentation-guided attention (Exp 3):** masking is applied "
                "identically in train/val/test in this run, so the Exp 3 test "
                "numbers are a valid comparison against Exp 1/Exp 2."
            )
        else:
            conclusion_1 = (
                "1. **Enhancement impact:** WPT→LMMSE→CLAHE pre-processing did **not** improve "
                "classification accuracy in these runs — Exp 2 (enhanced) scored below the "
                "raw-image baseline Exp 1. Enhancement improved CNR/boundary delineation for "
                "segmentation; its classification value is unproven."
            )
            conclusion_2 = (
                "2. **Segmentation-guided attention (Exp 3):** reported numbers are **invalid** — "
                "they come from pre-fix runs where masking was applied in training but not in "
                "validation. No conclusion can be drawn until Exp 3 is re-run with the fixed "
                "pipeline (identical masking in train/val/test)."
            )
        if xai_rows:
            conclusion_3 = (
                "3. **Explainability:** quantitative Grad-CAM localization was measured "
                "against held-out ground-truth masks (table in §2). IoU > 0.5 indicates "
                "the model's attention substantially overlaps the annotated tumor region."
            )
        else:
            conclusion_3 = (
                "3. **Explainability:** quantitative Grad-CAM localization was never run "
                '("data not yet available") — no spatial-alignment claim may be cited.'
            )
        if pmram_has_provenance:
            conclusion_4 = (
                "4. **Generalization:** PMRAM external validation was run with recorded "
                "checkpoint provenance (path + SHA-256 in "
                "results/pmram_external_validation.json) on the input distribution "
                "matching the evaluated model's training. Judge the gap in §3 on that basis."
            )
        else:
            conclusion_4 = (
                "4. **Generalization:** PMRAM external validation is encouraging but its "
                "provenance is unverified (data source recorded as folder-walk, BRISC "
                "reference mismatched). Re-run with recorded checkpoint provenance before "
                "citing."
            )

        report = f"""# PHASE II FINAL PROJECT REPORT
## MRI Image Enhancing and Tumor Detection
### An Explainable Deep Learning Framework for Brain Tumor Segmentation and Classification

> **Institution:** ITS Engineering College, AKTU  
> **Generated:** {ts}  
> **Dataset:** BRISC 2025 + PMRAM External Validation

{self._horizontal_rule()}

## Executive Summary

This Phase II report consolidates all three SRS-defined classification experiments, quantitative
Grad-CAM XAI localization analysis, and external generalization validation on the PMRAM dataset.
The segmentation-guided experiment (Exp 3) applies U-Net-derived soft attention weighting
(not hard RoI cropping) to guide the EfficientNetB2 classifier.

{provenance_note}

{self._horizontal_rule()}

## 1. Three-Experiment Classification Summary

{self._metric_table(all_rows, headers)}

{"![All Experiments Comparison](figures/model_comparison.png)" if fig_all else ""}

{"![Convergence Curves](figures/convergence_curves.png)" if fig_conv else ""}

{self._horizontal_rule()}

## 2. Grad-CAM Explainability — Quantitative Localization Analysis

Grad-CAM heatmaps (predicted class) were binarized at fixed activation thresholds
(0.3 / 0.5 / 0.7) and compared against ground-truth segmentation masks from the
held-out test split using pixel-level IoU and Dice metrics.

{self._metric_table(xai_rows, ["Experiment", "CAM Threshold", "Mean IoU", "Mean Dice", "Samples"]) if xai_rows else "*XAI localization data not yet available. Run `python -m explainability.run_localization_eval`.*"}

> **Interpretation:** IoU > 0.5 indicates the model's attention region substantially overlaps
> with the pathological region confirmed by the radiologist-annotated segmentation mask.

{self._horizontal_rule()}

## 3. External Generalization — PMRAM Validation

Zero-retraining inference was performed on the PMRAM dataset ({pmram_n} usable,
non-augmented brain MRI scans) to assess cross-dataset generalization.

| Metric | BRISC Internal | PMRAM External | Gap (Δ) |
|---|---|---|---|
| Accuracy | {self._pct(brisc_acc)} | {self._pct(pmram_acc)} | {self._pct(gen_gap.get("accuracy"))} |
| Macro F1 | {self._pct(brisc_m.get("macro_f1"))} | {self._pct(pmram_m.get("macro_f1"))} | {self._pct(gen_gap.get("macro_f1"))} |
| Precision | {self._pct(brisc_m.get("macro_precision"))} | {self._pct(pmram_m.get("macro_precision"))} | — |
| Recall | {self._pct(brisc_m.get("macro_recall"))} | {self._pct(pmram_m.get("macro_recall"))} | — |

{"![Generalization Gap](figures/generalization_gap.png)" if fig_gap else "*PMRAM validation data not yet available.*"}

{pmram_caveats}

{self._horizontal_rule()}

## 4. Conclusions

{conclusion_1}
{conclusion_2}
{conclusion_3}
{conclusion_4}

{self._horizontal_rule()}

## 5. Limitations & Future Work

- BRISC 2025 is limited to 4 tumor classes; multi-grade glioma sub-typing is planned.
- DICOM ingestion with native spatial resolution (voxel spacing) is a priority for clinical deployment.
- Prospective validation on local hospital PACS data is recommended before CE-marking submission.
- **Open data questions** (unresolvable without the raw datasets — verify on ingestion):
  patient-level grouping is not enforced (multiple slices per patient would leak
  across the image-level splits); whether the segmentation task contains
  `no_tumor` images is unconfirmed; classification folder-name casing on
  case-sensitive filesystems is assumed from the documented layout.

{self._horizontal_rule()}

*Report auto-generated by `evaluation/consolidate_reports.py` — MRI Image Enhancing and Tumor Detection*  
*ITS Engineering College · Supervisor: Mr. Manish Kumar Sharma · AKTU*
"""
        path = os.path.join(self.reports_dir, "PHASE_II_FINAL_REPORT.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Phase II report saved → {path}")

    def run(self):
        self.generate_phase1_report()
        self.generate_phase2_report()
        logger.info("All reports generated successfully.")


def main():
    consolidator = ReportConsolidator()
    consolidator.run()


if __name__ == "__main__":
    main()
