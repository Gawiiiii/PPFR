#!/usr/bin/env python3
"""Shared helpers for method-specific reconstruction benchmark scripts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["LFW", "CelebA", "AgeDB", "cplfw", "calfw"]
DEFAULT_METHODS = ["DuetFace", "ProFace", "PartialFace", "AdvFace", "MinusFace", "Ours"]
DEFAULT_ATTACKERS = ["gfpgan", "codeformer", "osdface"]

DATASET_ALIASES = {
    "lfw": "LFW",
    "celeba": "CelebA",
    "agedb": "AgeDB",
    "cplfw": "cplfw",
    "calfw": "calfw",
}

LFW_PAIR_FILES = {
    "DuetFace": REPO_ROOT / "datasets/LFW/txt/DuetFace_image_paths.txt",
    "ProFace": REPO_ROOT / "datasets/LFW/txt/PROFace_image_paths.txt",
    "PartialFace": REPO_ROOT / "datasets/LFW/txt/partialface_image_paths.txt",
    "AdvFace": REPO_ROOT / "datasets/LFW/txt/AdvFace_image_paths.txt",
    "MinusFace": REPO_ROOT / "datasets/LFW/txt/MinusFace_image_paths.txt",
    "Ours": REPO_ROOT / "datasets/LFW/txt/Ours_image_paths.txt",
}


def _method_name_candidates(method: str) -> List[str]:
    """Return candidate spellings used by different data dumps."""
    raw = method.strip()
    cands = [raw, raw.lower(), raw.upper(), raw.capitalize()]
    if raw.lower() == "proface":
        cands.extend(["PROFace", "ProFace_Blur"])
    if raw.lower() == "partialface":
        cands.extend(["partialface"])
    if raw.lower() == "fracface":
        cands.extend(["FracFace", "fracface", "FRACFACE"])
    # Preserve insertion order and drop duplicates.
    uniq: List[str] = []
    for item in cands:
        if item not in uniq:
            uniq.append(item)
    return uniq


def canonical_dataset(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_ALIASES[key]


def resolve_default_pair_file(dataset: str, method: str) -> Path:
    dataset = canonical_dataset(dataset)
    if dataset == "LFW" and method in LFW_PAIR_FILES:
        return LFW_PAIR_FILES[method]

    method_cands = _method_name_candidates(method)
    dataset_cands = [dataset, dataset.lower(), dataset.upper()]
    candidates: List[Path] = []

    if dataset == "LFW":
        txt_root = REPO_ROOT / "datasets" / "LFW" / "txt"
        for m in method_cands:
            candidates.append(txt_root / f"{m}_image_paths.txt")
            candidates.append(txt_root / f"{m}_pair_new.txt")

    home = Path.home()
    for d in dataset_cands:
        for m in method_cands:
            candidates.extend(
                [
                    Path(f"/home/ps/Public/YuanWei/OtherLab/{d}_{m}.txt"),
                    REPO_ROOT.parent / "FracFace" / f"{d}_{m}.txt",
                    home / f"{d}_{m}.txt",
                    home / "OtherLab" / f"{d}_{m}.txt",
                    home / "FracFace" / f"{d}_{m}.txt",
                    home / "fracface" / f"{d}_{m}.txt",
                ]
            )

    for cand in candidates:
        if cand.exists():
            return cand
    # Keep a deterministic fallback path for clearer errors upstream.
    return Path(f"/home/ps/Public/YuanWei/OtherLab/{dataset}_{method}.txt")


def read_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def write_pairs(path: Path, pairs: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def safe_link_or_copy(src: Path, dst: Path, copy_fallback: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        if not copy_fallback:
            raise
        shutil.copy2(src, dst)


def _save_resized_rgb(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize((size, size), Image.BILINEAR)
        im.save(dst)


def prepare_paired_folders(
    pair_file: Path,
    output_root: Path,
    max_pairs: int = 0,
    resize_to: int = 0,
    copy_files: bool = False,
) -> Dict[str, object]:
    """Create paired folders:
      - output_root/gt
      - output_root/lq

    Returns dict with keys:
      entries: List[Dict[str, str]]
      gt_dir: str
      lq_dir: str
    """

    pairs = read_pairs(pair_file)
    gt_dir = output_root / "gt"
    lq_dir = output_root / "lq"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, str]] = []
    for idx, (orig, prot) in enumerate(pairs):
        if max_pairs > 0 and idx >= max_pairs:
            break
        orig_p = Path(orig)
        prot_p = Path(prot)
        if not orig_p.exists() or not prot_p.exists():
            continue

        suffix = prot_p.suffix if prot_p.suffix else ".png"
        name = f"{prot_p.stem}__{short_hash(prot)}{suffix}"
        gt_dst = gt_dir / name
        lq_dst = lq_dir / name

        if resize_to > 0:
            _save_resized_rgb(orig_p, gt_dst, resize_to)
            _save_resized_rgb(prot_p, lq_dst, resize_to)
        else:
            if copy_files:
                if not gt_dst.exists():
                    shutil.copy2(orig_p, gt_dst)
                if not lq_dst.exists():
                    shutil.copy2(prot_p, lq_dst)
            else:
                safe_link_or_copy(orig_p, gt_dst)
                safe_link_or_copy(prot_p, lq_dst)

        entries.append(
            {
                "name": name,
                "original": str(orig_p),
                "protected": str(prot_p),
                "gt": str(gt_dst),
                "lq": str(lq_dst),
            }
        )

    return {
        "entries": entries,
        "gt_dir": str(gt_dir),
        "lq_dir": str(lq_dir),
    }


def run_command(
    cmd: Sequence[str],
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    log_file: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"\n[CMD] {' '.join(cmd)}\n")
            lf.flush()
            proc = subprocess.run(
                list(cmd),
                cwd=str(cwd),
                env=run_env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    else:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=run_env,
            text=True,
            check=False,
        )

    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")
    return proc


def find_latest_checkpoint(root: Path, suffixes: Sequence[str] = (".pth", ".pt", ".ckpt")) -> Optional[Path]:
    if not root.exists():
        return None
    cands: List[Path] = []
    for suffix in suffixes:
        cands.extend(root.rglob(f"*{suffix}"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def parse_kv_overrides(items: Sequence[str]) -> Dict[Tuple[str, str], Path]:
    """Parse `dataset:method=/abs/path.txt` list."""
    out: Dict[Tuple[str, str], Path] = {}
    for raw in items:
        if "=" not in raw or ":" not in raw.split("=", 1)[0]:
            raise ValueError(
                "Invalid override format. Use dataset:method=/abs/path/file.txt"
            )
        left, path_str = raw.split("=", 1)
        dataset_raw, method = left.split(":", 1)
        dataset = canonical_dataset(dataset_raw)
        out[(dataset, method)] = Path(path_str).expanduser().resolve()
    return out


def ensure_module_on_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
