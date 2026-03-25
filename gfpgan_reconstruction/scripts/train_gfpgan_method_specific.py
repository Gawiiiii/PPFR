#!/usr/bin/env python3
"""Train GFPGAN in method-specific setting: protected -> original."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_BIN_DEFAULT = "/home/ps/anaconda3/envs/cvpr/bin/python"
BASICSR_ROOT = REPO_ROOT / "third_party" / "BasicSR"
BASICSR_BUILD_CMD = (
    f"cd {BASICSR_ROOT} && BASICSR_EXT=True FORCE_CUDA=1 "
    f"{PYTHON_BIN_DEFAULT} setup.py build_ext --inplace"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from method_specific_common import (  # noqa: E402
    canonical_dataset,
    prepare_paired_folders,
    resolve_default_pair_file,
    run_command,
)


def _default_pair_file(pairs_root: Path, dataset: str, method: str, split: str) -> Path:
    return pairs_root / dataset / f"{method}_{split}.txt"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_gpu_ids(raw: str) -> List[int]:
    ids: List[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        if not text.isdigit():
            raise ValueError(f"Invalid --gpu-id value `{raw}`. Expected comma-separated GPU indexes.")
        ids.append(int(text))
    if not ids:
        raise ValueError("No GPU id provided. Use --gpu-id with at least one CUDA index.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicated GPU indexes in --gpu-id `{raw}`.")
    return ids


def _require_gpu_runtime(args: argparse.Namespace, visible_gpu_ids: List[int]) -> Dict[str, object]:
    if args.num_gpu <= 0:
        raise ValueError(f"--num-gpu must be > 0, got {args.num_gpu}.")
    if args.num_gpu > len(visible_gpu_ids):
        raise ValueError(
            f"--num-gpu={args.num_gpu} exceeds visible GPU ids `{args.gpu_id}` ({len(visible_gpu_ids)})."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GFPGAN training is GPU-only and cannot continue on CPU. "
            "Check NVIDIA driver/CUDA runtime and CUDA_VISIBLE_DEVICES."
        )
    visible_count = torch.cuda.device_count()
    if visible_count < args.num_gpu:
        raise RuntimeError(
            f"Only {visible_count} CUDA devices are visible after CUDA_VISIBLE_DEVICES={args.gpu_id}, "
            f"but --num-gpu={args.num_gpu}."
        )
    # Force a CUDA allocation so failures happen before expensive data preparation.
    torch.empty(1, device="cuda:0")
    names = [torch.cuda.get_device_name(i) for i in range(visible_count)]
    return {
        "cuda_available": True,
        "visible_cuda_count": visible_count,
        "visible_cuda_names": names,
        "requested_num_gpu": args.num_gpu,
        "requested_gpu_id": args.gpu_id,
    }


def _check_basicsr_cuda_extensions() -> Dict[str, str]:
    basicsr_root_str = str(BASICSR_ROOT)
    if basicsr_root_str not in sys.path:
        sys.path.insert(0, basicsr_root_str)

    up_module = importlib.import_module("basicsr.ops.upfirdn2d.upfirdn2d")
    fused_module = importlib.import_module("basicsr.ops.fused_act.fused_act")

    if not hasattr(up_module, "upfirdn2d_ext"):
        raise RuntimeError(
            "Missing CUDA extension `upfirdn2d_ext` for BasicSR. "
            f"Expected import path: {BASICSR_ROOT / 'basicsr/ops/upfirdn2d'}. "
            f"Build command: {BASICSR_BUILD_CMD}"
        )
    if not hasattr(fused_module, "fused_act_ext"):
        raise RuntimeError(
            "Missing CUDA extension `fused_act_ext` for BasicSR. "
            f"Expected import path: {BASICSR_ROOT / 'basicsr/ops/fused_act'}. "
            f"Build command: {BASICSR_BUILD_CMD}"
        )

    import torch

    x = torch.randn(1, 8, 8, 8, device="cuda:0")
    kernel = torch.randn(3, 3, device="cuda:0")
    bias = torch.zeros(8, device="cuda:0")
    _ = up_module.upfirdn2d(x, kernel)
    _ = fused_module.fused_leaky_relu(x, bias)

    return {
        "upfirdn2d_ext_file": str(up_module.upfirdn2d_ext.__file__),
        "fused_act_ext_file": str(fused_module.fused_act_ext.__file__),
    }


def _append_preflight_log(log_file: Path, lines: List[str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n[PRECHECK]\n")
        for line in lines:
            f.write(f"{line}\n")


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _stabilize_num_workers(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")
    requested = int(args.num_workers)
    effective = requested
    reason = ""

    # Python 3.13 + torch DataLoader multiprocessing is unstable in this env and can hard-stall training.
    if sys.version_info >= (3, 13) and requested > 0:
        effective = 0
        reason = (
            "Python 3.13 multiprocessing DataLoader instability detected; "
            "force --num-workers=0 to avoid resource_sharer deadlock."
        )
        print(f"[WARN] {reason} (requested={requested}, effective={effective})")

    args.requested_num_workers = requested
    args.num_workers = effective
    args.num_workers_adjust_reason = reason

# 生成basicsr训练yaml的字典
def build_options(
    template_path: Path,
    exp_root: Path,
    run_name: str,
    pretrained_g: Path,
    train_lq_dir: Path,
    train_gt_dir: Path,
    val_lq_dir: Path,
    val_gt_dir: Path,
    args: argparse.Namespace,
    has_val: bool,
) -> Dict:
    with template_path.open("r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    opt["name"] = run_name
    opt["num_gpu"] = args.num_gpu
    opt["manual_seed"] = args.seed

    net_g = opt.get("network_g", {})
    decoder_load_path = args.decoder_load_path.strip()
    if decoder_load_path:
        net_g["decoder_load_path"] = decoder_load_path
    else:
        existing_decoder_path = str(net_g.get("decoder_load_path", "")).strip()
        if existing_decoder_path and not Path(existing_decoder_path).exists():
            net_g["decoder_load_path"] = None
    net_g["fix_decoder"] = bool(args.fix_decoder)
    opt["network_g"] = net_g

    opt.setdefault("path", {})
    opt["path"]["experiments_root"] = str(exp_root)
    opt["path"]["pretrain_network_g"] = str(pretrained_g)
    opt["path"]["strict_load_g"] = False

    # 训练集配置 设置为成对图像训练 lq=受保护图 gt=原始图
    train_ds = {
        "name": f"{run_name}_train",
        "type": "PairedImageDataset",
        "dataroot_lq": str(train_lq_dir),
        "dataroot_gt": str(train_gt_dir),
        "io_backend": {"type": "disk"},
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "gt_size": args.train_size,
        "scale": 1,
        "use_hflip": args.use_flip,
        "use_rot": False,
        "num_worker_per_gpu": args.num_workers,
        "batch_size_per_gpu": args.batch_size,
        "dataset_enlarge_ratio": args.dataset_enlarge_ratio,
        "prefetch_mode": None,
    }
    # 验证集配置 
    val_ds = {
        "name": f"{run_name}_val",
        "type": "PairedImageDataset",
        "dataroot_lq": str(val_lq_dir),
        "dataroot_gt": str(val_gt_dir),
        "io_backend": {"type": "disk"},
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "scale": 1,
    }

    opt["datasets"] = {"train": train_ds}
    if has_val:
        opt["datasets"]["val"] = val_ds

    # 生成器 判别器学习率配置
    opt.setdefault("train", {})
    opt["train"]["total_iter"] = args.total_iter
    if "optim_g" in opt["train"]:
        opt["train"]["optim_g"]["lr"] = args.lr_g
    if "optim_d" in opt["train"]:
        opt["train"]["optim_d"]["lr"] = args.lr_d
    if "optim_component" in opt["train"]:
        opt["train"]["optim_component"]["lr"] = args.lr_d
    if not args.enable_perceptual:
        opt["train"]["perceptual_opt"] = None

    opt.setdefault("logger", {})
    opt["logger"]["print_freq"] = args.print_freq
    opt["logger"]["save_checkpoint_freq"] = args.save_checkpoint_freq
    opt["logger"]["use_tb_logger"] = args.use_tb_logger

    opt.setdefault("val", {})
    opt["val"]["val_freq"] = args.val_freq if has_val else int(1e12)
    opt["val"]["save_img"] = args.save_val_img
    # Always track both PSNR and SSIM for convergence and early-stopping decisions.
    val_metrics = opt["val"].get("metrics") or {}
    val_metrics.setdefault(
        "psnr",
        {
            "type": "calculate_psnr",
            "crop_border": 0,
            "test_y_channel": False,
        },
    )
    val_metrics.setdefault(
        "ssim",
        {
            "type": "calculate_ssim",
            "crop_border": 0,
            "test_y_channel": False,
        },
    )
    opt["val"]["metrics"] = val_metrics

    # early stop 逻辑
    early_metrics = [x.strip() for x in args.early_stop_metrics if x.strip()]
    if args.enable_early_stop and has_val and early_metrics:
        metric_min_delta = {m: 0.0 for m in early_metrics}
        if "psnr" in metric_min_delta:
            metric_min_delta["psnr"] = args.early_stop_min_delta_psnr
        if "ssim" in metric_min_delta:
            metric_min_delta["ssim"] = args.early_stop_min_delta_ssim
        opt["train"]["early_stop"] = {
            "enabled": True,
            "patience": int(args.early_stop_patience),
            "min_iter": int(args.early_stop_min_iter),
            "metrics": early_metrics,
            "metric_min_delta": metric_min_delta,
        }
    else:
        opt["train"]["early_stop"] = {"enabled": False}

    return opt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train method-specific GFPGAN checkpoint.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--pairs-root",
        default=str(REPO_ROOT / "method_specific_data" / "pairs"),
        help="Root that contains <dataset>/<method>_{train,val,test}.txt",
    )
    parser.add_argument("--train-pairs", default="")
    parser.add_argument("--val-pairs", default="")
    parser.add_argument(
        "--template",
        default=str(REPO_ROOT / "third_party/GFPGAN/options/train_gfpgan_v1_simple.yml"),
    )
    parser.add_argument(
        "--pretrained",
        default=str(REPO_ROOT / "third_party/GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth"),
    )
    parser.add_argument(
        "--decoder-load-path",
        default="",
        help=(
            "Optional StyleGAN2 decoder checkpoint path. Empty means auto: use template "
            "path when it exists, otherwise disable explicit decoder loading."
        ),
    )
    parser.add_argument(
        "--fix-decoder",
        action="store_true",
        help="Freeze StyleGAN2 decoder parameters during finetuning.",
    )
    parser.add_argument(
        "--ckpt-root",
        default="/home/ps/Public/YuanWei/OtherLab/method_specific_ckpts",
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "tmp" / "method_specific" / "gfpgan"),
    )
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--copy-files", action="store_true")

    parser.add_argument("--total-iter", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-enlarge-ratio", type=int, default=1)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--num-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-flip", action="store_true")
    parser.add_argument(
        "--enable-perceptual",
        action="store_true",
        help="Enable perceptual loss that requires VGG weights (may trigger online download).",
    )

    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--save-checkpoint-freq", type=int, default=2000)
    parser.add_argument("--val-freq", type=int, default=1000)
    parser.add_argument("--save-val-img", action="store_true")
    parser.add_argument("--use-tb-logger", action="store_true")
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="Disable validation-based early stop.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Stop when validation metrics do not improve for N consecutive validations.",
    )
    parser.add_argument(
        "--early-stop-min-iter",
        type=int,
        default=3000,
        help="Enable early-stop checks only after this iteration.",
    )
    parser.add_argument(
        "--early-stop-metrics",
        nargs="+",
        default=["psnr", "ssim"],
        help="Validation metrics to monitor for early-stop.",
    )
    parser.add_argument("--early-stop-min-delta-psnr", type=float, default=0.0)
    parser.add_argument("--early-stop-min-delta-ssim", type=float, default=0.0)

    parser.add_argument("--python", default=sys.executable, help="Python executable for training.")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--keep-training-artifacts",
        action="store_true",
        help="Keep intermediate BasicSR artifacts (models/training_states) under ckpt run dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.enable_early_stop = not args.disable_early_stop
    _stabilize_num_workers(args)

    visible_gpu_ids = _parse_gpu_ids(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["PYTHONUNBUFFERED"] = "1"

    dataset = canonical_dataset(args.dataset)
    method = args.method

    pairs_root = Path(args.pairs_root).expanduser().resolve()
    train_pairs = (
        Path(args.train_pairs).expanduser().resolve()
        if args.train_pairs
        else _default_pair_file(pairs_root, dataset, method, "train")
    )
    val_pairs = (
        Path(args.val_pairs).expanduser().resolve()
        if args.val_pairs
        else _default_pair_file(pairs_root, dataset, method, "val")
    )

    if not train_pairs.exists():
        # fallback to full pairs if split files are not prepared yet
        train_pairs = resolve_default_pair_file(dataset, method)
    if not train_pairs.exists():
        raise FileNotFoundError(f"Train pair file not found: {train_pairs}")

    if not val_pairs.exists():
        val_pairs = train_pairs

    template = Path(args.template).expanduser().resolve()
    pretrained = Path(args.pretrained).expanduser().resolve()
    if not template.exists():
        raise FileNotFoundError(f"Template yaml not found: {template}")
    if not pretrained.exists():
        raise FileNotFoundError(f"Pretrained GFPGAN checkpoint not found: {pretrained}")

    ckpt_root = Path(args.ckpt_root).expanduser().resolve() / "gfpgan" / dataset / method
    work_root = Path(args.work_root).expanduser().resolve() / dataset / method
    run_name = f"method_specific_gfpgan_{dataset}_{method}"
    log_file = ckpt_root / "run.log"

    gpu_report = _require_gpu_runtime(args, visible_gpu_ids)
    ext_report = _check_basicsr_cuda_extensions()
    _append_preflight_log(
        log_file,
        [
            f"CUDA_VISIBLE_DEVICES={args.gpu_id}",
            (
                f"num_gpu={args.num_gpu}, batch_size={args.batch_size}, "
                f"num_workers(requested/effective)={args.requested_num_workers}/{args.num_workers}"
            ),
            f"num_workers_adjust_reason={args.num_workers_adjust_reason or 'none'}",
            f"gpu_names={gpu_report['visible_cuda_names']}",
            f"upfirdn2d_ext={ext_report['upfirdn2d_ext_file']}",
            f"fused_act_ext={ext_report['fused_act_ext_file']}",
        ],
    )
    print(f"[PRECHECK] CUDA devices: {gpu_report['visible_cuda_names']}")
    print(f"[PRECHECK] upfirdn2d_ext: {ext_report['upfirdn2d_ext_file']}")
    print(f"[PRECHECK] fused_act_ext: {ext_report['fused_act_ext_file']}")

    train_data = prepare_paired_folders(
        pair_file=train_pairs,
        output_root=work_root / "data" / "train",
        max_pairs=args.max_train_pairs,
        resize_to=args.train_size,
        copy_files=args.copy_files,
    )
    val_data = prepare_paired_folders(
        pair_file=val_pairs,
        output_root=work_root / "data" / "val",
        max_pairs=args.max_val_pairs,
        resize_to=args.train_size,
        copy_files=args.copy_files,
    )

    has_val = len(val_data["entries"]) > 0
    options = build_options(
        template_path=template,
        exp_root=ckpt_root,
        run_name=run_name,
        pretrained_g=pretrained,
        train_lq_dir=Path(train_data["lq_dir"]),
        train_gt_dir=Path(train_data["gt_dir"]),
        val_lq_dir=Path(val_data["lq_dir"]),
        val_gt_dir=Path(val_data["gt_dir"]),
        args=args,
        has_val=has_val,
    )

    opt_path = work_root / "configs" / f"{run_name}.yml"
    _ensure_parent(opt_path)
    with opt_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(options, f, sort_keys=False)

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    env = {
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(BASICSR_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else ""),
    }

    cmd = [
        args.python,
        str(REPO_ROOT / "third_party/GFPGAN/gfpgan/train.py"),
        "-opt",
        str(opt_path),
        "--launcher",
        "none",
    ]

    status = "dry_run"
    if args.dry_run:
        print("[DRY-RUN]", " ".join(cmd))
    else:
        run_command(cmd, cwd=REPO_ROOT / "third_party/GFPGAN", env=env, log_file=log_file, check=True)
        status = "ok"

    model_dir = ckpt_root / run_name / "models"
    final_ckpt = None
    if model_dir.exists():
        preferred = model_dir / "net_g_latest.pth"
        if preferred.exists():
            final_ckpt = preferred
        else:
            cands = sorted(model_dir.glob("net_g_*.pth"))
            if cands:
                final_ckpt = cands[-1]

    export_ckpt = ckpt_root / f"gfpgan_{dataset}_{method}.pth"
    if final_ckpt and final_ckpt.exists() and not args.dry_run:
        shutil.copy2(final_ckpt, export_ckpt)

    cleanup_removed: Dict[str, int] = {}
    if not args.keep_training_artifacts and not args.dry_run:
        exp_run_dir = ckpt_root / run_name
        training_states_dir = exp_run_dir / "training_states"
        if training_states_dir.exists():
            cleanup_removed["training_states"] = _dir_size_bytes(training_states_dir)
            shutil.rmtree(training_states_dir, ignore_errors=False)
        # Only prune model snapshots after export checkpoint is safely available.
        if export_ckpt.exists():
            model_dir = exp_run_dir / "models"
            if model_dir.exists():
                cleanup_removed["models"] = _dir_size_bytes(model_dir)
                shutil.rmtree(model_dir, ignore_errors=False)

    meta = {
        "dataset": dataset,
        "method": method,
        "status": status,
        "train_pairs": str(train_pairs),
        "val_pairs": str(val_pairs),
        "train_entries": len(train_data["entries"]),
        "val_entries": len(val_data["entries"]),
        "train_size": args.train_size,
        "total_iter": args.total_iter,
        "num_gpu": args.num_gpu,
        "gpu_id": args.gpu_id,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "requested_num_workers": args.requested_num_workers,
        "num_workers_adjust_reason": args.num_workers_adjust_reason,
        "save_checkpoint_freq": args.save_checkpoint_freq,
        "cuda_runtime": gpu_report,
        "cuda_extensions": ext_report,
        "keep_training_artifacts": args.keep_training_artifacts,
        "cleanup_removed_bytes": cleanup_removed,
        "early_stop": {
            "enabled": args.enable_early_stop,
            "patience": args.early_stop_patience,
            "min_iter": args.early_stop_min_iter,
            "metrics": list(args.early_stop_metrics),
            "min_delta_psnr": args.early_stop_min_delta_psnr,
            "min_delta_ssim": args.early_stop_min_delta_ssim,
        },
        "option_yaml": str(opt_path),
        "log_file": str(log_file),
        "export_ckpt": str(export_ckpt if export_ckpt.exists() else ""),
        "updated_at": int(time.time()),
    }
    meta_path = ckpt_root / "train_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved training meta: {meta_path}")
    if export_ckpt.exists():
        print(f"Exported checkpoint: {export_ckpt}")
    if cleanup_removed:
        removed_mb = {k: round(v / 1024 / 1024, 2) for k, v in cleanup_removed.items()}
        print(f"Pruned training artifacts (MB): {removed_mb}")


if __name__ == "__main__":
    main()
