# GFPGAN Reconstruction Package

This folder packages the latest GFPGAN reconstruction-attack scripts and the actual formal-run data subset used in the `2025CVPR` project.

Source provenance:
- Scripts were copied from `/home/ps/Public/YuanWei/2025CVPR`.
- Pair/meta/reference files were copied from `/home/ps/Public/YuanWei/OtherLab/method_specific_full_runs/gfpgan_formal_earlystop_20260311_182159`.

Included scripts:
- `scripts/prepare_method_specific_pairs.py`: prepares method-specific `train/val/test` pair splits.
- `scripts/method_specific_common.py`: shared helpers for pair loading, folder export, and subprocess execution.
- `scripts/train_gfpgan_method_specific.py`: GFPGAN training wrapper for `protected -> original`.
- `scripts/run_method_specific_gfpgan_full_train_test.py`: full formal pipeline that prepares splits, trains GFPGAN checkpoints, tests them, and records status.
- `scripts/run_method_specific_recon_benchmark.py`: unified reconstruction benchmark runner used for GFPGAN testing.
- `third_party/GFPGAN/inference_gfpgan.py`: GFPGAN inference entry used by the benchmark.
- `third_party/GFPGAN/gfpgan/train.py`: GFPGAN training entry invoked by the wrapper.
- `third_party/GFPGAN/options/train_gfpgan_v1_simple.yml`: GFPGAN training template used to generate per-run YAML files.

Included data:
- `data/pairs/`: formal-run pair files for `LFW`, `CelebA`, `AgeDB`, `cplfw`, and `calfw`.
- `data/meta/`: dataset split metadata exported during pair preparation.
- `data/assets/`: the exact image subset referenced by the exported pair files.
- `reference_run/`: `train_test_status.csv` and summary CSVs from the formal GFPGAN run.

Important path note:
- Pair files were rewritten from machine-specific absolute paths to repository-relative paths rooted at `gfpgan_reconstruction/data/assets/...`.
- Run the scripts from the repository root if you want those relative paths to resolve directly.

Dependency note:
- This package preserves the latest project-side wrappers, pair data, and directly invoked GFPGAN entry files.
- End-to-end execution still requires the full runtime environment used by the original project, including the rest of GFPGAN, BasicSR, and related Python dependencies.
