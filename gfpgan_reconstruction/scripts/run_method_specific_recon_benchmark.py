#!/usr/bin/env python3
"""Run method-specific reconstruction benchmark across attackers/datasets/methods."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from method_specific_common import (  # noqa: E402
    DEFAULT_ATTACKERS,
    DEFAULT_DATASETS,
    DEFAULT_METHODS,
    canonical_dataset,
    find_latest_checkpoint,
    parse_kv_overrides,
    read_pairs,
    resolve_default_pair_file,
    run_command,
    short_hash,
    write_pairs,
)
from run_diffusion_recon_attack import eval_psnr_ssim  # noqa: E402


def parse_ckpt_overrides(items: Sequence[str]) -> Dict[Tuple[str, str, str], Path]:
    out: Dict[Tuple[str, str, str], Path] = {}
    for raw in items:
        if "=" not in raw or raw.count(":") < 2:
            raise ValueError(
                "Invalid --ckpt-override format. Use attacker:dataset:method=/abs/path/to/ckpt"
            )
        left, path_str = raw.split("=", 1)
        attacker, dataset_raw, method = left.split(":", 2)
        dataset = canonical_dataset(dataset_raw)
        out[(attacker.lower(), dataset, method)] = Path(path_str).expanduser().resolve()
    return out


def resolve_pair_file(
    dataset: str,
    method: str,
    split: str,
    pairs_root: Path,
    overrides: Dict[Tuple[str, str], Path],
) -> Path:
    split_file = pairs_root / dataset / f"{method}_{split}.txt"
    if split_file.exists():
        return split_file
    key = (dataset, method)
    if key in overrides:
        return overrides[key]
    return resolve_default_pair_file(dataset, method)


def resolve_checkpoint(
    attacker: str,
    dataset: str,
    method: str,
    ckpt_root: Path,
    overrides: Dict[Tuple[str, str, str], Path],
) -> Optional[Path]:
    key = (attacker, dataset, method)
    if key in overrides:
        path = overrides[key]
        return path if path.exists() else None

    root = ckpt_root / attacker / dataset / method
    if not root.exists():
        return None

    # Prefer standardized export filenames when present.
    preferred_names = [
        f"{attacker}_{dataset}_{method}.pth",
        f"{attacker}_{dataset}_{method}.pt",
        f"{attacker}_{dataset}_{method}.ckpt",
        f"{attacker}_lora_{dataset}_{method}.ckpt",
    ]
    for name in preferred_names:
        p = root / name
        if p.exists():
            return p

    # OSDFace may save a directory that contains LoRA files.
    if attacker == "osdface":
        required = ["pytorch_lora_weights.safetensors", "embedding_change_weights.pth"]
        for d in sorted([x for x in root.iterdir() if x.is_dir()]):
            if all((d / r).exists() for r in required):
                return d
        if all((root / r).exists() for r in required):
            return root

    return find_latest_checkpoint(root)


def prepare_inputs(
    pairs: Sequence[Tuple[str, str]],
    input_dir: Path,
    max_pairs: int,
    copy_files: bool,
) -> List[Dict[str, str]]:
    input_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, str]] = []
    for idx, (orig, prot) in enumerate(pairs):
        if max_pairs > 0 and idx >= max_pairs:
            break
        prot_path = Path(prot)
        if not prot_path.exists() or not Path(orig).exists():
            continue
        suffix = prot_path.suffix if prot_path.suffix else ".png"
        name = f"{prot_path.stem}__{short_hash(prot)}{suffix}"
        dst = input_dir / name
        if not dst.exists():
            if copy_files:
                shutil.copy2(prot_path, dst)
            else:
                try:
                    os.symlink(prot_path, dst)
                except OSError:
                    shutil.copy2(prot_path, dst)
        entries.append({"original": orig, "protected": prot, "input_name": name})
    return entries


def build_recon_pairs(
    attacker: str,
    entries: Sequence[Dict[str, str]],
    output_dir: Path,
    recon_list: Path,
) -> Tuple[int, int]:
    pairs: List[Tuple[str, str]] = []
    valid = 0
    for entry in entries:
        orig = entry["original"]
        name = entry["input_name"]
        stem = Path(name).stem

        if attacker == "gfpgan":
            cands = [
                output_dir / "restored_faces" / f"{stem}_00.png",
                output_dir / "restored_faces" / f"{stem}.png",
                output_dir / "restored_imgs" / name,
                output_dir / "restored_imgs" / f"{stem}.png",
            ]
        elif attacker == "codeformer":
            cands = [
                output_dir / "restored_faces" / f"{stem}.png",
                output_dir / "restored_faces" / name,
                output_dir / "final_results" / f"{stem}.png",
                output_dir / "final_results" / name,
            ]
        else:
            cands = [
                output_dir / "recon" / name,
                output_dir / "recon" / f"{stem}.png",
                output_dir / name,
                output_dir / f"{stem}.png",
            ]

        chosen = cands[0]
        for cand in cands:
            if cand.exists():
                chosen = cand
                break
        if chosen.exists():
            valid += 1
        pairs.append((orig, str(chosen)))

    write_pairs(recon_list, pairs)
    return len(pairs), valid


def _python_prefix(conda_env: str, python_exe: str) -> List[str]:
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python"]
    return [python_exe]


@contextmanager
def _patched_weight(expected_path: Path, custom_ckpt: Optional[Path], backup_dir: Path):
    """Temporarily replace a fixed-path weight file and restore it afterwards."""
    if custom_ckpt is None:
        yield
        return
    if expected_path.exists() and custom_ckpt.exists():
        try:
            if expected_path.resolve() == custom_ckpt.resolve():
                yield
                return
        except OSError:
            pass

    backup_dir.mkdir(parents=True, exist_ok=True)
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    backup = backup_dir / f"{expected_path.name}.bak"
    had_original = expected_path.exists()
    if had_original:
        shutil.copy2(expected_path, backup)

    shutil.copy2(custom_ckpt, expected_path)
    try:
        yield
    finally:
        if expected_path.exists():
            expected_path.unlink()
        if had_original and backup.exists():
            shutil.move(str(backup), str(expected_path))


def run_gfpgan(
    entries: Sequence[Dict[str, str]],
    input_dir: Path,
    output_dir: Path,
    ckpt: Optional[Path],
    combo_dir: Path,
    args: argparse.Namespace,
    log_file: Path,
    env: Dict[str, str],
) -> int:
    if not entries:
        return 0

    root = REPO_ROOT / "third_party/GFPGAN"
    output_dir.mkdir(parents=True, exist_ok=True)
    gfpgan_version = "1.4"
    model_file = "GFPGANv1.4.pth"
    if ckpt is not None and ckpt.exists():
        try:
            obj = torch.load(str(ckpt), map_location="cpu")
            if isinstance(obj, dict):
                state = obj.get("params_ema") or obj.get("params") or obj
                if isinstance(state, dict) and any(
                    str(k).startswith("conv_body_first.0.") for k in state.keys()
                ):
                    # Method-specific training uses GFPGANv1 (channel_multiplier=1).
                    gfpgan_version = "1"
                    model_file = "GFPGANv1.pth"
        except Exception:
            # Keep default version on metadata probing failures; real load errors
            # are reported by GFPGAN inference logs.
            pass

    expected_weight = root / "experiments/pretrained_models" / model_file

    cmd = _python_prefix(args.conda_env, args.python) + [
        str(root / "inference_gfpgan.py"),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-v",
        gfpgan_version,
        "-s",
        "1",
        "--aligned",
        "--bg_upsampler",
        "none",
        "--ext",
        "png",
        "-w",
        str(args.gfpgan_weight),
    ]

    with _patched_weight(expected_weight, ckpt, combo_dir / "weight_backups"):
        proc = run_command(cmd, cwd=root, env=env, log_file=log_file, check=False)
    return proc.returncode


def run_codeformer(
    input_dir: Path,
    output_dir: Path,
    ckpt: Optional[Path],
    combo_dir: Path,
    args: argparse.Namespace,
    log_file: Path,
    env: Dict[str, str],
) -> int:
    root = REPO_ROOT / "third_party/CodeFormer"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_weight = root / "weights/CodeFormer/codeformer.pth"

    cmd = _python_prefix(args.conda_env, args.python) + [
        str(root / "inference_codeformer.py"),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-w",
        str(args.codeformer_w),
        "--has_aligned",
        "--bg_upsampler",
        "None",
    ]

    with _patched_weight(expected_weight, ckpt, combo_dir / "weight_backups"):
        proc = run_command(cmd, cwd=root, env=env, log_file=log_file, check=False)
    return proc.returncode


def run_osdface(
    input_dir: Path,
    output_dir: Path,
    ckpt: Optional[Path],
    args: argparse.Namespace,
    log_file: Path,
    env: Dict[str, str],
) -> int:
    root = REPO_ROOT / "third_party/diffusion_attacks/OSDFace"
    output_dir.mkdir(parents=True, exist_ok=True)

    if ckpt is None:
        return 1
    ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent

    raw_gpu_tokens = [x.strip() for x in str(args.gpu_id).replace(",", " ").split() if x.strip()]
    # CUDA_VISIBLE_DEVICES remaps visible devices to local indices [0..N-1].
    # For example, gpu_id="1" means only one visible device (local cuda:0).
    if raw_gpu_tokens:
        gpu_ids = [str(i) for i in range(len(raw_gpu_tokens))]
    else:
        gpu_ids = ["0"]

    cmd = _python_prefix(args.conda_env, args.python) + [
        str(root / "infer.py"),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir / "recon"),
        "--pretrained_model_name_or_path",
        args.osdface_pretrained,
        "--ckpt_path",
        str(ckpt_dir),
        "--img_encoder_weight",
        args.osdface_img_encoder,
        "--mixed_precision",
        args.osdface_precision,
        "--seed",
        str(args.seed),
        "--gpu_ids",
        *gpu_ids,
    ]

    proc = run_command(cmd, cwd=root, env=env, log_file=log_file, check=False)
    return proc.returncode


def write_summary(rows: List[Dict[str, str]], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "attacker",
                "split",
                "pairs",
                "valid_pairs",
                "psnr",
                "ssim",
                "lpips",
                "mse",
                "status",
                "elapsed_sec",
                "pair_file",
                "ckpt",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(rows: List[Dict[str, str]], attackers: Sequence[str], matrix_csv: Path) -> None:
    keyed: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        if key not in keyed:
            keyed[key] = {"dataset": row["dataset"], "method": row["method"]}
        atk = row["attacker"]
        keyed[key][f"{atk}_status"] = row.get("status", "")
        keyed[key][f"{atk}_psnr"] = row.get("psnr", "")
        keyed[key][f"{atk}_ssim"] = row.get("ssim", "")
        keyed[key][f"{atk}_lpips"] = row.get("lpips", "")
        keyed[key][f"{atk}_mse"] = row.get("mse", "")

    fieldnames = ["dataset", "method"]
    for atk in attackers:
        fieldnames.extend(
            [f"{atk}_status", f"{atk}_psnr", f"{atk}_ssim", f"{atk}_lpips", f"{atk}_mse"]
        )

    matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in sorted(keyed.items()):
            writer.writerow(row)


def parse_eval_size(side: int) -> Optional[Tuple[int, int]]:
    if side <= 0:
        return None
    return side, side


def main() -> None:
    parser = argparse.ArgumentParser(description="Run method-specific recon benchmark.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pairs-root", default=str(REPO_ROOT / "method_specific_data" / "pairs"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])

    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--attackers", nargs="+", default=DEFAULT_ATTACKERS)

    parser.add_argument(
        "--ckpt-root",
        default="/home/ps/Public/YuanWei/OtherLab/method_specific_ckpts",
    )
    parser.add_argument("--pair-file-override", action="append", default=[])
    parser.add_argument("--ckpt-override", action="append", default=[])

    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--copy-inputs", action="store_true")
    parser.add_argument("--eval-size", type=int, default=112)

    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conda-env", default="")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--gfpgan-weight", type=float, default=0.5)
    parser.add_argument("--codeformer-w", type=float, default=0.5)

    parser.add_argument(
        "--osdface-pretrained",
        default="stabilityai/stable-diffusion-2-1-base",
    )
    parser.add_argument(
        "--osdface-img-encoder",
        default=str(REPO_ROOT / "third_party/diffusion_attacks/OSDFace/pretrained/associate_2.ckpt"),
    )
    parser.add_argument("--osdface-precision", choices=["fp16", "fp32"], default="fp16")

    parser.add_argument("--skip-missing-pairs", action="store_true")
    parser.add_argument("--skip-missing-ckpt", action="store_true")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail immediately if CUDA is unavailable for inference/evaluation.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the whole benchmark once any dataset/method/attacker combo fails.",
    )
    args = parser.parse_args()

    datasets = [canonical_dataset(x) for x in args.datasets]
    methods = list(args.methods)
    attackers = [x.lower() for x in args.attackers]

    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    pairs_root = Path(args.pairs_root).expanduser().resolve()
    ckpt_root = Path(args.ckpt_root).expanduser().resolve()

    pair_overrides = parse_kv_overrides(args.pair_file_override)
    ckpt_overrides = parse_ckpt_overrides(args.ckpt_override)

    summary_csv = run_root / "summary_metrics.csv"
    matrix_csv = run_root / "benchmark_matrix.csv"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env = {
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "PYTHONUNBUFFERED": "1",
    }

    if args.require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available while --require-gpu is set. "
                "Method-specific benchmark refuses CPU fallback."
            )
        if torch.cuda.device_count() < 1:
            raise RuntimeError(
                f"No CUDA device visible after CUDA_VISIBLE_DEVICES={args.gpu_id}."
            )
        torch.empty(1, device="cuda:0")
        eval_device = torch.device("cuda")
    else:
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_size = parse_eval_size(args.eval_size)

    rows: List[Dict[str, str]] = []

    def record_row(row: Dict[str, str], fail_message: str = "") -> None:
        rows.append(row)
        write_summary(rows, summary_csv)
        if fail_message and args.fail_fast:
            raise RuntimeError(fail_message)

    for dataset in datasets:
        for method in methods:
            pair_file = resolve_pair_file(dataset, method, args.split, pairs_root, pair_overrides)
            for attacker in attackers:
                t0 = time.time()
                combo_dir = run_root / "combos" / dataset / method / attacker
                combo_dir.mkdir(parents=True, exist_ok=True)
                log_file = combo_dir / "run.log"
                output_dir = combo_dir / "outputs"

                row: Dict[str, str] = {
                    "dataset": dataset,
                    "method": method,
                    "attacker": attacker,
                    "split": args.split,
                    "pairs": "0",
                    "valid_pairs": "0",
                    "psnr": "",
                    "ssim": "",
                    "lpips": "",
                    "mse": "",
                    "status": "init",
                    "elapsed_sec": "",
                    "pair_file": str(pair_file),
                    "ckpt": "",
                    "run_dir": str(combo_dir),
                }

                if not pair_file.exists():
                    row["status"] = "missing_pair_file"
                    row["elapsed_sec"] = f"{time.time() - t0:.2f}"
                    fail_msg = f"Missing pair file for {dataset}/{method}: {pair_file}"
                    if args.skip_missing_pairs:
                        record_row(row)
                    else:
                        record_row(row, fail_msg)
                    continue

                pairs = read_pairs(pair_file)
                entries = prepare_inputs(
                    pairs,
                    combo_dir / "inputs",
                    max_pairs=args.max_pairs,
                    copy_files=args.copy_inputs,
                )
                row["pairs"] = str(len(entries))
                if not entries:
                    row["status"] = "empty_pairs"
                    row["elapsed_sec"] = f"{time.time() - t0:.2f}"
                    record_row(
                        row,
                        f"Pair file has no usable entries for {dataset}/{method}/{attacker}: {pair_file}",
                    )
                    continue

                ckpt = resolve_checkpoint(attacker, dataset, method, ckpt_root, ckpt_overrides)
                row["ckpt"] = str(ckpt) if ckpt else ""
                if ckpt is None or not ckpt.exists():
                    row["status"] = "missing_ckpt"
                    row["elapsed_sec"] = f"{time.time() - t0:.2f}"
                    fail_msg = f"Missing checkpoint for {attacker}/{dataset}/{method} under {ckpt_root}"
                    if args.skip_missing_ckpt:
                        record_row(row)
                        continue
                    record_row(row, fail_msg)
                    continue

                rc = 1
                if attacker == "gfpgan":
                    rc = run_gfpgan(
                        entries=entries,
                        input_dir=combo_dir / "inputs",
                        output_dir=output_dir,
                        ckpt=ckpt,
                        combo_dir=combo_dir,
                        args=args,
                        log_file=log_file,
                        env=env,
                    )
                elif attacker == "codeformer":
                    rc = run_codeformer(
                        input_dir=combo_dir / "inputs",
                        output_dir=output_dir,
                        ckpt=ckpt,
                        combo_dir=combo_dir,
                        args=args,
                        log_file=log_file,
                        env=env,
                    )
                elif attacker == "osdface":
                    rc = run_osdface(
                        input_dir=combo_dir / "inputs",
                        output_dir=output_dir,
                        ckpt=ckpt,
                        args=args,
                        log_file=log_file,
                        env=env,
                    )
                else:
                    row["status"] = "unsupported_attacker"
                    row["elapsed_sec"] = f"{time.time() - t0:.2f}"
                    record_row(row, f"Unsupported attacker: {attacker}")
                    continue

                if rc != 0:
                    row["status"] = f"attack_failed_rc{rc}"
                    row["elapsed_sec"] = f"{time.time() - t0:.2f}"
                    record_row(
                        row,
                        f"Attack run failed with rc={rc} for {attacker}/{dataset}/{method}. Check {log_file}",
                    )
                    continue

                recon_list = combo_dir / "recon_pairs.txt"
                total_pairs, valid_pairs = build_recon_pairs(attacker, entries, output_dir, recon_list)
                row["pairs"] = str(total_pairs)
                row["valid_pairs"] = str(valid_pairs)

                if valid_pairs == 0:
                    row["status"] = "no_recon_output"
                    row["elapsed_sec"] = f"{time.time() - t0:.2f}"
                    record_row(
                        row,
                        f"No reconstruction outputs for {attacker}/{dataset}/{method}. Check {log_file}",
                    )
                    continue

                metrics = eval_psnr_ssim(recon_list, eval_device, eval_size=eval_size)
                row["psnr"] = f"{metrics['psnr']:.6f}"
                row["ssim"] = f"{metrics['ssim']:.6f}"
                row["lpips"] = f"{metrics['lpips']:.6f}"
                row["mse"] = f"{metrics['mse']:.6f}"
                row["valid_pairs"] = str(int(metrics.get("valid_pairs", float(valid_pairs))))
                row["status"] = "ok"
                row["elapsed_sec"] = f"{time.time() - t0:.2f}"

                metrics_json = combo_dir / "metrics.json"
                metrics_json.write_text(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "method": method,
                            "attacker": attacker,
                            "split": args.split,
                            "pair_file": str(pair_file),
                            "recon_pairs": str(recon_list),
                            "checkpoint": str(ckpt),
                            "metrics": metrics,
                            "elapsed_sec": float(row["elapsed_sec"]),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                record_row(row)

    write_summary(rows, summary_csv)
    write_matrix(rows, attackers, matrix_csv)
    print(f"Saved summary: {summary_csv}")
    print(f"Saved matrix: {matrix_csv}")


if __name__ == "__main__":
    main()
