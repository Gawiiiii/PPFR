#!/usr/bin/env python3
"""Prepare method-specific paired data with identity-level train/val/test splits.

Outputs:
- <output_root>/pairs/<dataset>/<method>_train.txt
- <output_root>/pairs/<dataset>/<method>_val.txt
- <output_root>/pairs/<dataset>/<method>_test.txt
- <output_root>/meta/<dataset>/id_split.json
- <output_root>/meta/prepare_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["LFW", "CelebA", "AgeDB", "cplfw", "calfw"]
DEFAULT_METHODS = ["DuetFace", "ProFace", "PartialFace", "AdvFace", "MinusFace", "Ours", "FracFace"]
SPLITS = ("train", "val", "test")

LFW_PAIR_FILES = {
    "DuetFace": REPO_ROOT / "datasets/LFW/txt/DuetFace_image_paths.txt",
    "ProFace": REPO_ROOT / "datasets/LFW/txt/PROFace_image_paths.txt",
    "PartialFace": REPO_ROOT / "datasets/LFW/txt/partialface_image_paths.txt",
    "AdvFace": REPO_ROOT / "datasets/LFW/txt/AdvFace_image_paths.txt",
    "MinusFace": REPO_ROOT / "datasets/LFW/txt/MinusFace_image_paths.txt",
    "Ours": REPO_ROOT / "datasets/LFW/txt/Ours_image_paths.txt",
}

DATASET_ALIASES = {
    "lfw": "LFW",
    "celeba": "CelebA",
    "agedb": "AgeDB",
    "cplfw": "cplfw",
    "calfw": "calfw",
}


@dataclass
class PairRow:
    original: str
    protected: str
    identity: str


def canonical_dataset(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_ALIASES[key]


def _method_name_candidates(method: str) -> List[str]:
    raw = method.strip()
    cands = [raw, raw.lower(), raw.upper(), raw.capitalize()]
    if raw.lower() == "proface":
        cands.extend(["PROFace", "ProFace_Blur"])
    if raw.lower() == "partialface":
        cands.extend(["partialface"])
    if raw.lower() == "fracface":
        cands.extend(["FracFace", "fracface", "FRACFACE"])
    uniq: List[str] = []
    for item in cands:
        if item not in uniq:
            uniq.append(item)
    return uniq


def resolve_default_pair_file(dataset: str, method: str) -> Path:
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
    return Path(f"/home/ps/Public/YuanWei/OtherLab/{dataset}_{method}.txt")


def read_pairs(path: Path, dataset: str) -> List[PairRow]:
    rows: List[PairRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            original, protected = parts[0], parts[1]
            identity = extract_identity(original, dataset)
            rows.append(PairRow(original=original, protected=protected, identity=identity))
    return rows


def extract_identity(original_path: str, dataset: str) -> str:
    p = Path(original_path)
    parent = p.parent.name

    if dataset == "AgeDB":
        m = re.match(r"^\d+_(.+)_\d+_[mMfF]$", parent)
        if m:
            return m.group(1)
        parts = parent.split("_")
        if len(parts) >= 4 and parts[0].isdigit():
            return "_".join(parts[1:-2])
        return parent

    if dataset in {"LFW", "cplfw", "calfw"}:
        # Strip image index suffix for identity-level splitting.
        # Examples: John_Doe_0001 -> John_Doe, Joseph_Biden_2 -> Joseph_Biden
        trimmed = re.sub(r"_\d{1,4}$", "", parent)
        return trimmed if trimmed else parent

    return parent


def parse_ratio(value: str) -> Tuple[float, float, float]:
    raw = value.replace(" ", "")
    parts = raw.split(",")
    if len(parts) != 3:
        raise ValueError("--split-ratio must be 'train,val,test', e.g. 0.8,0.1,0.1")
    train_r, val_r, test_r = [float(x) for x in parts]
    total = train_r + val_r + test_r
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return train_r / total, val_r / total, test_r / total


def _stable_dataset_seed(base_seed: int, dataset: str) -> int:
    h = int(hashlib.sha1(dataset.encode("utf-8")).hexdigest()[:8], 16)
    return base_seed + (h % 1_000_000)


def split_identities(
    identities: Iterable[str],
    ratio: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[str]]:
    ids = sorted(set(identities))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    train_r, val_r, test_r = ratio
    raw_counts = [n * train_r, n * val_r, n * test_r]
    counts = [int(x) for x in raw_counts]
    remain = n - sum(counts)
    order = sorted(range(3), key=lambda i: raw_counts[i] - counts[i], reverse=True)
    for i in range(remain):
        counts[order[i % 3]] += 1

    n_train, n_val, n_test = counts
    # Keep all splits non-empty when possible.
    if n >= 3:
        if n_val == 0:
            n_val = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_test -= 1
        if n_test == 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_val -= 1

    i1 = n_train
    i2 = n_train + n_val
    return {
        "train": ids[:i1],
        "val": ids[i1:i2],
        "test": ids[i2:i2 + n_test],
    }


def parse_overrides(items: Sequence[str]) -> Dict[Tuple[str, str], Path]:
    overrides: Dict[Tuple[str, str], Path] = {}
    for raw in items:
        if "=" not in raw or ":" not in raw.split("=", 1)[0]:
            raise ValueError(
                "Invalid override format. Use --pair-file-override dataset:method=/abs/path/file.txt"
            )
        left, path_str = raw.split("=", 1)
        dataset_raw, method = left.split(":", 1)
        dataset = canonical_dataset(dataset_raw)
        overrides[(dataset, method)] = Path(path_str).expanduser().resolve()
    return overrides


def write_pair_list(path: Path, rows: Sequence[PairRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row.original} {row.protected}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare method-specific train/val/test pair files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to process.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to process.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "method_specific_data"),
        help="Output root for pairs/meta.",
    )
    parser.add_argument(
        "--split-ratio",
        default="0.8,0.1,0.1",
        help="train,val,test ratio, e.g. 0.8,0.1,0.1",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--id-set-mode",
        choices=["intersection", "union", "reference"],
        default="intersection",
        help="How to choose identity universe shared across methods.",
    )
    parser.add_argument(
        "--reference-method",
        default="DuetFace",
        help="Used when --id-set-mode=reference.",
    )
    parser.add_argument(
        "--pair-file-override",
        action="append",
        default=[],
        help="Override pair file with dataset:method=/abs/path.txt (repeatable).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing pair files (except methods listed in --allow-missing-method).",
    )
    parser.add_argument(
        "--allow-missing-method",
        action="append",
        default=["AdvFace"],
        help="Method names allowed to be missing without failing.",
    )
    args = parser.parse_args()

    datasets = [canonical_dataset(x) for x in args.datasets]
    methods = list(args.methods)
    ratio = parse_ratio(args.split_ratio)
    output_root = Path(args.output_root).expanduser().resolve()
    overrides = parse_overrides(args.pair_file_override)
    allow_missing = set(args.allow_missing_method)

    rows_for_summary: List[Dict[str, object]] = []

    for dataset in datasets:
        method_to_rows: Dict[str, List[PairRow]] = {}
        method_to_ids: Dict[str, set] = {}
        method_meta: Dict[str, Dict[str, object]] = {}

        for method in methods:
            pair_file = overrides.get((dataset, method), resolve_default_pair_file(dataset, method))
            meta_item: Dict[str, object] = {
                "pair_file": str(pair_file),
                "status": "init",
                "pairs_total": 0,
                "identity_total": 0,
                "identity_shared": 0,
                "split_pairs": {"train": 0, "val": 0, "test": 0},
            }

            if not pair_file.exists():
                meta_item["status"] = "missing_pair_file"
                if args.strict and method not in allow_missing:
                    raise FileNotFoundError(f"Missing pair file: {pair_file}")
                method_meta[method] = meta_item
                rows_for_summary.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": meta_item["status"],
                        "pair_file": str(pair_file),
                        "pairs_total": 0,
                        "identity_total": 0,
                        "identity_shared": 0,
                        "train_pairs": 0,
                        "val_pairs": 0,
                        "test_pairs": 0,
                    }
                )
                continue

            rows = read_pairs(pair_file, dataset)
            if not rows:
                meta_item["status"] = "empty_pair_file"
                method_meta[method] = meta_item
                rows_for_summary.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": meta_item["status"],
                        "pair_file": str(pair_file),
                        "pairs_total": 0,
                        "identity_total": 0,
                        "identity_shared": 0,
                        "train_pairs": 0,
                        "val_pairs": 0,
                        "test_pairs": 0,
                    }
                )
                continue

            identity_set = {r.identity for r in rows}
            method_to_rows[method] = rows
            method_to_ids[method] = identity_set
            meta_item["status"] = "ok"
            meta_item["pairs_total"] = len(rows)
            meta_item["identity_total"] = len(identity_set)
            method_meta[method] = meta_item

        available_methods = sorted(method_to_rows.keys())
        if not available_methods:
            dataset_meta = {
                "dataset": dataset,
                "seed": args.seed,
                "split_ratio": {"train": ratio[0], "val": ratio[1], "test": ratio[2]},
                "id_set_mode": args.id_set_mode,
                "status": "no_available_methods",
                "shared_identity_count": 0,
                "split_identity_counts": {"train": 0, "val": 0, "test": 0},
                "split_identities": {"train": [], "val": [], "test": []},
                "methods": method_meta,
            }
            meta_path = output_root / "meta" / dataset / "id_split.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
            continue

        if args.id_set_mode == "intersection":
            shared_ids = set.intersection(*(method_to_ids[m] for m in available_methods))
        elif args.id_set_mode == "union":
            shared_ids = set.union(*(method_to_ids[m] for m in available_methods))
        else:
            ref_method = args.reference_method
            if ref_method not in method_to_ids:
                ref_method = available_methods[0]
            shared_ids = set(method_to_ids[ref_method])

        fallback_note = ""
        if not shared_ids:
            shared_ids = set.union(*(method_to_ids[m] for m in available_methods))
            fallback_note = "shared_identity_set_empty_fallback_to_union"

        split_ids = split_identities(
            shared_ids,
            ratio=ratio,
            seed=_stable_dataset_seed(args.seed, dataset),
        )
        split_id_sets = {k: set(v) for k, v in split_ids.items()}

        for method in methods:
            pair_dir = output_root / "pairs" / dataset
            split_files = {
                split: pair_dir / f"{method}_{split}.txt"
                for split in SPLITS
            }
            meta_item = method_meta[method]
            if method not in method_to_rows:
                for split in SPLITS:
                    write_pair_list(split_files[split], [])
                continue

            scoped_rows = [r for r in method_to_rows[method] if r.identity in shared_ids]
            meta_item["identity_shared"] = len({r.identity for r in scoped_rows})

            for split in SPLITS:
                split_rows = [r for r in scoped_rows if r.identity in split_id_sets[split]]
                write_pair_list(split_files[split], split_rows)
                meta_item["split_pairs"][split] = len(split_rows)

            rows_for_summary.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "status": meta_item["status"],
                    "pair_file": meta_item["pair_file"],
                    "pairs_total": meta_item["pairs_total"],
                    "identity_total": meta_item["identity_total"],
                    "identity_shared": meta_item["identity_shared"],
                    "train_pairs": meta_item["split_pairs"]["train"],
                    "val_pairs": meta_item["split_pairs"]["val"],
                    "test_pairs": meta_item["split_pairs"]["test"],
                }
            )

        dataset_meta = {
            "dataset": dataset,
            "seed": args.seed,
            "split_ratio": {"train": ratio[0], "val": ratio[1], "test": ratio[2]},
            "id_set_mode": args.id_set_mode,
            "status": "ok",
            "shared_identity_count": len(shared_ids),
            "split_identity_counts": {k: len(v) for k, v in split_ids.items()},
            "split_identities": split_ids,
            "fallback_note": fallback_note,
            "methods": method_meta,
        }

        meta_path = output_root / "meta" / dataset / "id_split.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")

    summary_csv = output_root / "meta" / "prepare_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "status",
                "pair_file",
                "pairs_total",
                "identity_total",
                "identity_shared",
                "train_pairs",
                "val_pairs",
                "test_pairs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_for_summary)

    print(f"Prepared method-specific pairs under: {output_root}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
