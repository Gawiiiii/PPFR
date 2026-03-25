#!/usr/bin/env python3
"""Strict method-specific full train+test runner for GFPGAN only.

Pipeline for each dataset/method combo:
1) train on *_train split
2) test on *_test split using the freshly trained checkpoint
3) record PSNR/SSIM and ID attack success rates (ASR)

The script fails fast on any error and refuses CPU fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method_specific_common import canonical_dataset  # noqa: E402
from run_diffusion_recon_attack import eval_reconstruction  # noqa: E402

DEFAULT_DATASETS = ["LFW", "CelebA", "AgeDB", "cplfw", "calfw"]
DEFAULT_METHODS = ["DuetFace", "ProFace", "MinusFace", "Ours", "AdvFace", "FracFace"]
DEFAULT_OPTIONAL_METHODS = ["AdvFace", "FracFace"]
DEFAULT_FALLBACK_SPLIT_ROOTS = [
    "/home/ps/Public/YuanWei/OtherLab/method_specific_full_runs/pair_data/pairs",
    "/home/ps/Public/YuanWei/OtherLab/method_specific_full_runs/full_20260306_030533/pair_data/pairs",
]
ASR_MODEL_ORDER = ["Mobilenet", "resnet", "IR", "ArcFace", "FaceNet", "CosFace"]
STATUS_FIELDS = [
    "dataset",
    "method",
    "attacker",
    "train_status",
    "test_status",
    "round_status",
    "error",
    "train_pairs_file",
    "test_pairs_file",
    "train_pairs",
    "test_pairs",
    "valid_pairs",
    "train_log",
    "test_log",
    "train_meta",
    "train_ckpt",
    "gpu_id",
    "gpu_names",
    "psnr",
    "ssim",
    "asr_avg",
    "asr_Mobilenet",
    "asr_resnet",
    "asr_IR",
    "asr_ArcFace",
    "asr_FaceNet",
    "asr_CosFace",
    "benchmark_run_dir",
    "updated_at",
]


def _run_logged(cmd: Sequence[str], cwd: Path, log_file: Path, env: Dict[str, str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n[CMD] {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed rc={proc.returncode}: {' '.join(cmd)}")


def _pair_count(path: Path) -> int:
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c


def _ensure_status_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()


def _append_status(path: Path, row: Dict[str, str]) -> None:
    _ensure_status_csv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writerow(row)


def _load_done_keys(status_csv: Path) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not status_csv.exists():
        return done
    with status_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("round_status") in {"ok", "skipped"}:
                dataset = row.get("dataset", "").strip()
                method = row.get("method", "").strip()
                if dataset and method:
                    done.add((dataset, method))
    return done


def _load_latest_rows(status_csv: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    latest: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not status_csv.exists():
        return latest
    with status_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset", "").strip()
            method = row.get("method", "").strip()
            if not dataset or not method:
                continue
            latest[(dataset, method)] = row
    return latest


def _find_resumable_ckpt(dataset: str, method: str, ckpt_root: Path, prior_row: Dict[str, str]) -> Path | None:
    prior_ckpt = prior_row.get("train_ckpt", "").strip()
    if prior_row.get("train_status") == "ok" and prior_ckpt:
        p = Path(prior_ckpt)
        if p.exists():
            return p

    meta_path = ckpt_root / "gfpgan" / dataset / method / "train_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("status", "")).lower() == "ok":
                export_ckpt = str(meta.get("export_ckpt", "")).strip()
                if export_ckpt:
                    p = Path(export_ckpt)
                    if p.exists():
                        return p
        except Exception:
            return None
    fallback = ckpt_root / "gfpgan" / dataset / method / f"gfpgan_{dataset}_{method}.pth"
    if fallback.exists():
        return fallback
    return None


def _read_single_benchmark_row(summary_csv: Path) -> Dict[str, str]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Benchmark summary missing: {summary_csv}")
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected 1 benchmark row in {summary_csv}, got {len(rows)}")
    return rows[0]


def _find_ckpt_from_meta(meta_path: Path, dataset: str, method: str, ckpt_root: Path) -> Path:
    if not meta_path.exists():
        raise FileNotFoundError(f"Training meta missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    export_ckpt = str(meta.get("export_ckpt", "")).strip()
    if export_ckpt:
        p = Path(export_ckpt)
        if p.exists():
            return p
    fallback = ckpt_root / "gfpgan" / dataset / method / f"gfpgan_{dataset}_{method}.pth"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Trained checkpoint not found for gfpgan/{dataset}/{method}. "
        f"Checked meta export_ckpt and fallback {fallback}"
    )


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[none]"
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total} ({ratio * 100:.1f}%)"


def _resolve_split_with_fallback(
    pair_root: Path,
    dataset: str,
    method: str,
    split: str,
    fallback_roots: Sequence[Path],
) -> Path:
    target = pair_root / "pairs" / dataset / f"{method}_{split}.txt"
    if target.exists() and _pair_count(target) > 0:
        return target
    for root in fallback_roots:
        cand = root / dataset / f"{method}_{split}.txt"
        if cand.exists():
            # Prefer non-empty fallback split files when multiple roots are provided.
            if _pair_count(cand) <= 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, target)
            return target
    return target


def _summarize(status_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    with status_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("round_status") == "ok":
                rows.append(row)

    detail_csv = output_dir / "gfpgan_dataset_method_detail.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    method_stats: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        method = row["method"]
        method_stats.setdefault(
            method,
            {"psnr": [], "ssim": [], "asr_avg": [], "valid_pairs": []},
        )
        method_stats[method]["psnr"].append(_safe_float(row.get("psnr", "")))
        method_stats[method]["ssim"].append(_safe_float(row.get("ssim", "")))
        method_stats[method]["asr_avg"].append(_safe_float(row.get("asr_avg", "")))
        method_stats[method]["valid_pairs"].append(_safe_float(row.get("valid_pairs", "")))

    method_csv = output_dir / "gfpgan_method_summary.csv"
    with method_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "count", "psnr_mean", "ssim_mean", "asr_avg_mean", "valid_pairs_mean"],
        )
        writer.writeheader()
        for method in DEFAULT_METHODS:
            vals = method_stats.get(method)
            if not vals:
                writer.writerow(
                    {
                        "method": method,
                        "count": 0,
                        "psnr_mean": "",
                        "ssim_mean": "",
                        "asr_avg_mean": "",
                        "valid_pairs_mean": "",
                    }
                )
                continue
            writer.writerow(
                {
                    "method": method,
                    "count": len(vals["psnr"]),
                    "psnr_mean": f"{mean(vals['psnr']):.6f}",
                    "ssim_mean": f"{mean(vals['ssim']):.6f}",
                    "asr_avg_mean": f"{mean(vals['asr_avg']):.6f}",
                    "valid_pairs_mean": f"{mean(vals['valid_pairs']):.2f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict full train and test using GFPGAN.")
    parser.add_argument("--python", default="/home/ps/anaconda3/envs/cvpr/bin/python")
    parser.add_argument("--run-tag", default=f"gfpgan_full_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--run-root",
        default="",
        help="Default: /home/ps/Public/YuanWei/OtherLab/method_specific_full_runs/<run-tag>",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--optional-methods", nargs="+", default=DEFAULT_OPTIONAL_METHODS)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--num-gpu", type=int, default=1)
    parser.add_argument("--total-iter", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--dataset-enlarge-ratio", type=int, default=1)
    parser.add_argument(
        "--save-checkpoint-freq",
        type=int,
        default=10000,
        help="GFPGAN checkpoint/save-state frequency during training.",
    )
    parser.add_argument(
        "--keep-training-artifacts",
        action="store_true",
        help="Keep intermediate BasicSR artifacts (models/training_states) for each combo.",
    )
    parser.add_argument("--eval-size", type=int, default=112)
    parser.add_argument("--gfpgan-weight", type=float, default=0.5)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-iter", type=int, default=3000)
    parser.add_argument("--early-stop-metrics", nargs="+", default=["psnr", "ssim"])
    parser.add_argument("--early-stop-min-delta-psnr", type=float, default=0.0)
    parser.add_argument("--early-stop-min-delta-ssim", type=float, default=0.0)
    parser.add_argument(
        "--split-fallback-root",
        action="append",
        default=DEFAULT_FALLBACK_SPLIT_ROOTS,
        help="Fallback roots that already contain <dataset>/<method>_{train,val,test}.txt",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    datasets = [canonical_dataset(x) for x in args.datasets]
    methods = list(args.methods)
    optional_methods = set(args.optional_methods)
    fallback_roots = [Path(x).expanduser().resolve() for x in args.split_fallback_root]
    for m in methods:
        if m not in DEFAULT_METHODS:
            raise ValueError(f"Unsupported method for this runner: {m}")

    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
    else:
        run_root = Path(f"/home/ps/Public/YuanWei/OtherLab/method_specific_full_runs/{args.run_tag}")

    pair_root = run_root / "pair_data"
    ckpt_root = run_root / "ckpts"
    work_root = run_root / "work"
    log_root = run_root / "logs"
    bench_root = run_root / "benchmark_rounds"
    summary_root = run_root / "summary"
    status_csv = run_root / "train_test_status.csv"

    for d in [pair_root, ckpt_root, work_root, log_root, bench_root, summary_root]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This pipeline is GPU-only and forbids CPU fallback."
        )
    if torch.cuda.device_count() < args.num_gpu:
        raise RuntimeError(
            f"Visible CUDA devices={torch.cuda.device_count()} < --num-gpu={args.num_gpu} "
            f"after CUDA_VISIBLE_DEVICES={args.gpu_id}"
        )
    torch.empty(1, device="cuda:0")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["PYTHONUNBUFFERED"] = "1"

    prepare_log = log_root / "prepare_pairs.log"
    prepare_cmd = [
        args.python,
        str(REPO_ROOT / "scripts/prepare_method_specific_pairs.py"),
        "--output-root",
        str(pair_root),
        "--id-set-mode",
        "intersection",
        "--datasets",
        *datasets,
        "--methods",
        *methods,
    ]
    for m in sorted(optional_methods):
        prepare_cmd.extend(["--allow-missing-method", m])
    _run_logged(prepare_cmd, cwd=REPO_ROOT, log_file=prepare_log, env=env)

    _ensure_status_csv(status_csv)
    done_keys = _load_done_keys(status_csv) if args.resume else set()
    latest_rows = _load_latest_rows(status_csv) if args.resume else {}
    total_combos = len(datasets) * len(methods)
    finished_combos = len(done_keys)
    print(f"[PROGRESS] { _progress_bar(finished_combos, total_combos) }")

    for dataset in datasets:
        for method in methods:
            if (dataset, method) in done_keys:
                continue
            combo_idx = finished_combos + 1
            print(f"[ROUND] ({combo_idx}/{total_combos}) {dataset}/{method}")

            prior_row = latest_rows.get((dataset, method), {})
            row: Dict[str, str] = {k: "" for k in STATUS_FIELDS}
            row.update(
                {
                    "dataset": dataset,
                    "method": method,
                    "attacker": "gfpgan",
                    "train_status": "failed",
                    "test_status": "failed",
                    "round_status": "failed",
                    "gpu_id": args.gpu_id,
                    "gpu_names": json.dumps(gpu_names, ensure_ascii=True),
                    "updated_at": str(int(time.time())),
                }
            )

            try:
                pair_train = _resolve_split_with_fallback(pair_root, dataset, method, "train", fallback_roots)
                pair_val = _resolve_split_with_fallback(pair_root, dataset, method, "val", fallback_roots)
                pair_test = _resolve_split_with_fallback(pair_root, dataset, method, "test", fallback_roots)
                missing = [p for p in [pair_train, pair_val, pair_test] if not p.exists()]
                if missing:
                    if method in optional_methods:
                        row["train_status"] = "skipped"
                        row["test_status"] = "skipped"
                        row["round_status"] = "skipped"
                        row["error"] = f"Optional method missing split files: {[str(p) for p in missing]}"
                        row["updated_at"] = str(int(time.time()))
                        _append_status(status_csv, row)
                        finished_combos += 1
                        print(
                            f"[SKIP] {dataset}/{method}: missing optional split files. "
                            f"{_progress_bar(finished_combos, total_combos)}"
                        )
                        continue
                    raise FileNotFoundError(f"Missing split pair file(s): {missing}")

                train_pairs = _pair_count(pair_train)
                test_pairs = _pair_count(pair_test)
                if train_pairs <= 0 or test_pairs <= 0:
                    if method in optional_methods:
                        row["train_status"] = "skipped"
                        row["test_status"] = "skipped"
                        row["round_status"] = "skipped"
                        row["error"] = (
                            f"Optional method has empty splits: train_pairs={train_pairs}, "
                            f"test_pairs={test_pairs}, files=({pair_train}, {pair_test})"
                        )
                        row["train_pairs_file"] = str(pair_train)
                        row["test_pairs_file"] = str(pair_test)
                        row["train_pairs"] = str(train_pairs)
                        row["test_pairs"] = str(test_pairs)
                        row["updated_at"] = str(int(time.time()))
                        _append_status(status_csv, row)
                        finished_combos += 1
                        print(
                            f"[SKIP] {dataset}/{method}: empty optional splits. "
                            f"{_progress_bar(finished_combos, total_combos)}"
                        )
                        continue
                    if train_pairs <= 0:
                        raise RuntimeError(f"Empty training pair file: {pair_train}")
                    raise RuntimeError(f"Empty test pair file: {pair_test}")

                row["train_pairs_file"] = str(pair_train)
                row["test_pairs_file"] = str(pair_test)
                row["train_pairs"] = str(train_pairs)
                row["test_pairs"] = str(test_pairs)

                meta_path = ckpt_root / "gfpgan" / dataset / method / "train_meta.json"
                row["train_meta"] = str(meta_path)
                train_log = log_root / f"train_gfpgan_{dataset}_{method}.log"
                row["train_log"] = str(train_log)
                train_ckpt = _find_resumable_ckpt(dataset, method, ckpt_root, prior_row if args.resume else {})
                if train_ckpt is None:
                    train_cmd = [
                        args.python,
                        str(REPO_ROOT / "scripts/train_gfpgan_method_specific.py"),
                        "--dataset",
                        dataset,
                        "--method",
                        method,
                        "--pairs-root",
                        str(pair_root / "pairs"),
                        "--train-pairs",
                        str(pair_train),
                        "--val-pairs",
                        str(pair_val),
                        "--python",
                        args.python,
                        "--gpu-id",
                        args.gpu_id,
                        "--num-gpu",
                        str(args.num_gpu),
                        "--ckpt-root",
                        str(ckpt_root),
                        "--work-root",
                        str(work_root),
                        "--total-iter",
                        str(args.total_iter),
                        "--batch-size",
                        str(args.batch_size),
                        "--num-workers",
                        str(args.num_workers),
                        "--dataset-enlarge-ratio",
                        str(args.dataset_enlarge_ratio),
                        "--save-checkpoint-freq",
                        str(args.save_checkpoint_freq),
                    ]
                    if args.keep_training_artifacts:
                        train_cmd.append("--keep-training-artifacts")
                    if args.disable_early_stop:
                        train_cmd.append("--disable-early-stop")
                    else:
                        train_cmd.extend(
                            [
                                "--early-stop-patience",
                                str(args.early_stop_patience),
                                "--early-stop-min-iter",
                                str(args.early_stop_min_iter),
                                "--early-stop-metrics",
                                *args.early_stop_metrics,
                                "--early-stop-min-delta-psnr",
                                str(args.early_stop_min_delta_psnr),
                                "--early-stop-min-delta-ssim",
                                str(args.early_stop_min_delta_ssim),
                            ]
                        )
                    _run_logged(train_cmd, cwd=REPO_ROOT, log_file=train_log, env=env)
                    train_ckpt = _find_ckpt_from_meta(meta_path, dataset, method, ckpt_root)
                else:
                    print(f"[RESUME] Reusing trained checkpoint for {dataset}/{method}: {train_ckpt}")

                row["train_status"] = "ok"
                row["train_ckpt"] = str(train_ckpt)
                row["test_status"] = "pending"
                row["round_status"] = "running"
                row["error"] = ""
                row["updated_at"] = str(int(time.time()))
                _append_status(status_csv, row)
                latest_rows[(dataset, method)] = dict(row)

                test_run_root = bench_root / dataset / method
                test_log = log_root / f"test_gfpgan_{dataset}_{method}.log"
                row["test_log"] = str(test_log)
                ckpt_override = f"gfpgan:{dataset}:{method}={train_ckpt}"
                test_cmd = [
                    args.python,
                    str(REPO_ROOT / "scripts/run_method_specific_recon_benchmark.py"),
                    "--run-root",
                    str(test_run_root),
                    "--pairs-root",
                    str(pair_root / "pairs"),
                    "--split",
                    "test",
                    "--datasets",
                    dataset,
                    "--methods",
                    method,
                    "--attackers",
                    "gfpgan",
                    "--ckpt-root",
                    str(ckpt_root),
                    "--ckpt-override",
                    ckpt_override,
                    "--python",
                    args.python,
                    "--gpu-id",
                    args.gpu_id,
                    "--eval-size",
                    str(args.eval_size),
                    "--gfpgan-weight",
                    str(args.gfpgan_weight),
                    "--require-gpu",
                    "--fail-fast",
                ]
                if args.max_test_pairs > 0:
                    test_cmd.extend(["--max-pairs", str(args.max_test_pairs)])
                _run_logged(test_cmd, cwd=REPO_ROOT, log_file=test_log, env=env)
                row["test_status"] = "ok"

                bench_row = _read_single_benchmark_row(test_run_root / "summary_metrics.csv")
                if bench_row.get("status") != "ok":
                    raise RuntimeError(
                        f"Benchmark status is not ok for {dataset}/{method}: {bench_row.get('status', '')}"
                    )

                row["psnr"] = bench_row.get("psnr", "")
                row["ssim"] = bench_row.get("ssim", "")
                row["valid_pairs"] = bench_row.get("valid_pairs", "")
                run_dir = Path(bench_row["run_dir"])
                row["benchmark_run_dir"] = str(run_dir)

                recon_pairs = run_dir / "recon_pairs.txt"
                if not recon_pairs.exists():
                    raise FileNotFoundError(f"Missing recon_pairs.txt: {recon_pairs}")

                valid_pairs = int(float(bench_row.get("valid_pairs", "0") or "0"))
                if valid_pairs <= 0:
                    raise RuntimeError(f"Invalid valid_pairs={valid_pairs} for {dataset}/{method}")

                asr_success = eval_reconstruction(recon_pairs, torch.device("cuda"))
                asr_rates: Dict[str, float] = {}
                for name in ASR_MODEL_ORDER:
                    success = int(asr_success.get(name, 0))
                    asr_rates[name] = success / valid_pairs

                row["asr_Mobilenet"] = f"{asr_rates['Mobilenet']:.6f}"
                row["asr_resnet"] = f"{asr_rates['resnet']:.6f}"
                row["asr_IR"] = f"{asr_rates['IR']:.6f}"
                row["asr_ArcFace"] = f"{asr_rates['ArcFace']:.6f}"
                row["asr_FaceNet"] = f"{asr_rates['FaceNet']:.6f}"
                row["asr_CosFace"] = f"{asr_rates['CosFace']:.6f}"
                row["asr_avg"] = f"{mean(asr_rates.values()):.6f}"

                asr_json = run_dir / "attack_success_rates.json"
                asr_json.write_text(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "method": method,
                            "attacker": "gfpgan",
                            "valid_pairs": valid_pairs,
                            "success_counts": asr_success,
                            "success_rates": asr_rates,
                            "asr_avg": float(row["asr_avg"]),
                            "checkpoint": str(train_ckpt),
                            "gpu_id": args.gpu_id,
                            "gpu_names": gpu_names,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                row["round_status"] = "ok"
                row["error"] = ""
                row["updated_at"] = str(int(time.time()))
                _append_status(status_csv, row)
                finished_combos += 1
                print(f"[OK] {dataset}/{method} { _progress_bar(finished_combos, total_combos) }")
            except Exception as e:
                row["error"] = str(e)
                row["updated_at"] = str(int(time.time()))
                _append_status(status_csv, row)
                raise

    _summarize(status_csv, summary_root)
    print(f"[DONE] GFPGAN full train+test finished. Run root: {run_root}")
    print(f"[DONE] Status CSV: {status_csv}")
    print(f"[DONE] Summary dir: {summary_root}")


if __name__ == "__main__":
    main()
