# Appearance Meets Geometry

**Semantic segmentation of ancient fortification facades from fused orthophoto and normal-map input.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

This repository holds the code and the evaluation record for the paper

> Nils Schnorr¹, Thomas Leimkühler² — *Appearance Meets Geometry: Deep Learning for Semantic
> Segmentation of Archaeological Fortification Masonry*
> ¹ Saarland University, Institute of Classical Archaeology · ² Max Planck Institute for Informatics
> Preprint, dataset and trained models: [10.5281/zenodo.21507975](https://doi.org/10.5281/zenodo.21507975)
> Presented at CAA 2025, Athens, Session 9

<!-- The concept DOI ("Cite all versions?" on the Zenodo record) always resolves to the newest
     version and is the better link here; swap it in when convenient. -->

## Abstract

Modern archaeological practice faces a critical bottleneck: while photogrammetric surveys
routinely generate terabytes of high-resolution 3D data from ancient fortifications, manual
interpretation methods remain time-consuming and subjective, leaving much documentation
underexploited. This paper presents a novel machine-learning workflow that bridges the gap
between high-throughput data acquisition and efficient analysis by leveraging both appearance
and geometric information from 3D photogrammetric models for automated masonry semantic
segmentation.

Our approach transforms 3D documentation into a dual-layer format combining high-resolution
orthomosaics with geometry-based normal maps derived from heightmap data. We trained
convolutional neural networks on annotated datasets from Carian fortification walls currently
spanning approximately 700 m across the ancient sites of Halicarnassus and Cedreae. The training
dataset comprises 1,876 m² of wall facade with 5,560 individually annotated stones across
several masonry classes.

Comparative evaluation of appearance-only, geometry-only, and combined models shows consistent
performance advantages for the integrated approach. In an ensemble evaluation over five
independent training runs per variant, the combined model improved mean stone IoU by 10% over
the geometry-only and 4% over the appearance-only model, with this ordering consistent across
all individual runs. These results confirm that visual and geometric features provide
complementary information essential for robust archaeological analysis. The automated approach
enables processing of kilometer-long fortifications in minutes of computation, compared to the
weeks of manual drawing and classification that traditional methods require. This transforms
archaeological documentation from selective sampling to comprehensive analysis of entire
defensive systems. This work represents a concrete step toward realizing photogrammetry's
analytical potential, enabling systematic comparative studies across sites and periods at
unprecedented scales.

*Keywords:* semantic segmentation, photogrammetry, normal maps, fortification architecture,
deep learning, machine learning, masonry classification, 3D documentation, Carian fortifications

## What this is

A multi-channel U-Net classifies masonry type per pixel from photogrammetric documentation.
Three input variants are trained and evaluated under one identical protocol: **geometry-only**
(3 channels, normal map), **appearance-only** (4 channels, RGB + alpha) and **fused**
(7 channels, RGB + alpha + normal map). Averaged over the four held-out test walls the fused
variant reaches the highest mean stone IoU — 0.583, against 0.560 appearance-only and 0.530
geometry-only. That ordering holds for the aggregate in every individual run; it does not hold
wall by wall, where Wall 4 favours the appearance-only variant. The per-wall figures are given
in full in the CSVs referenced in §8.

---

## 1. What is and is not in this repository

**Tracked here:**

| | |
|---|---|
| `amg_pipeline/` | the pipeline itself — one importable Python package, no logic in notebooks |
| `01_image_preparation_for_ML/` | dataset construction from orthophotos, DEMs and COCO annotations |
| `05_orchestration/` | notebooks that drive the package (config in, runs out) |
| `04_segmentation_evaluation/` | figure scripts and single-wall utilities |
| `experiments/` | the evaluation record: every manifest and metric CSV behind the paper |
| `Archive/` | the original v1 notebooks, kept for provenance; superseded, not maintained |

**Not tracked** (excluded by `.gitignore`, see §9 for where to get them): source orthophotos,
DEMs, annotations, generated masks and normal maps, training tiles, model checkpoints
(`*.pth`), and the full-resolution prediction rasters. The repository is therefore small; the
data deposit is separate.

Everything in `experiments/` is **derived output kept as the numerical record**. Do not edit it
by hand. Every number in the paper traces to a file there.

```
AppearanceMeetsGeometry/
├── amg_pipeline/                     # source of truth — all pipeline logic
│   ├── config.py                     # RunConfig: one dataclass carries a whole run
│   ├── paths.py                      # every output path derived from the run_id
│   ├── data.py  model.py  train.py   # tiles, MultiUNet, training loop
│   ├── segment.py  evaluate.py       # sliding-window inference, MC-ROI metrics
│   ├── sweep.py  ensemble.py         # variant × run driver, mean-softmax ensembling
│   ├── gapmetrics.py  boundary.py    # gap IoU, stone separation, Boundary IoU
│   ├── confidence.py                 # per-pixel top-1 confidence of the ensemble
│   ├── stress.py  perturb.py         # lighting-robustness stress test
│   ├── augment.py  history.py        # photometric augmentation, training-curve export
│   └── verify.py                     # architecture / checkpoint equivalence checks
├── 01_image_preparation_for_ML/
│   └── image_preparation_pipeline.ipynb
├── 05_orchestration/
│   ├── orchestrator.ipynb            # train → segment → evaluate sweeps
│   ├── final_numbers.ipynb           # candidate comparison + the paper's ensemble
│   ├── gap_metrics.ipynb             # gap / stone-separation evaluation
│   ├── stress_boundary_stage0.ipynb  # robustness + Boundary IoU
│   └── training_curves.ipynb         # curves from the checkpoints' embedded history
├── 04_segmentation_evaluation/
│   ├── confidence_heatmap.py         # confidence raster → figure
│   ├── confidence_probe_readout.py   # per-pixel probability readout figure
│   ├── run_probe_readout.ipynb
│   └── manual_single_wall_segmentation.ipynb   # standalone; not part of the paper
├── experiments/                      # manifests and metric CSVs (the record)
├── Archive/                          # original v1 notebooks
├── requirements.txt
└── LICENSE
```

---

## 2. The pipeline in one view

```
orthophoto (PNG, RGBA)  ─┐
DEM / heightmap (GeoTIFF)─┤  01_image_preparation_for_ML
COCO polygon annotations ─┘   ├─ masks from COCO
                              ├─ horizontal-flip augmentation
                              ├─ normal maps  (Mikkelsen + RANSAC yaw normalisation)
                              ├─ hold out the four test walls   ← automatic, step 5.5
                              ├─ Sobol tiling (1280 px, coverage 1.6)
                              └─ drop empty tiles
                                        │
                                        ▼   2,149 training tiles per modality
                              05_orchestration/orchestrator.ipynb
                              amg_pipeline:  train → segment → evaluate
                              3ch / 4ch / 7ch  ×  5 runs each
                                        │
                                        ▼   experiments/<name>/manifest.csv
                              05_orchestration/final_numbers.ipynb
                              mean-softmax ensemble of the 5 runs per variant
                                        │
                                        ▼   experiments/v2_baseline_ens/manifest.csv
                              gap / stone · Boundary IoU · confidence · stress
```

Two properties hold throughout and are worth knowing before running anything:

- **Every output path is derived from a `run_id`** (`{channels}ch_run{N}`) under
  `experiments/<experiment_name>/`. Runs cannot silently overwrite each other, and an
  experiment name is the unit of isolation — one screened variable, one experiment.
- **Every stage is skip-if-exists.** An interrupted sweep is resumed by running the cell
  again. `force=True` recomputes.

---

## 3. Environment

```bash
conda create -n amg python=3.11.9
conda activate amg

# GPU build; use the CPU wheels if you only intend to run the evaluation stages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Training and inference need CUDA in practice (they fall back to CPU, slowly). Everything
downstream of the saved rasters — ROI metrics, gap and stone metrics, Boundary IoU, the
figure scripts — is CPU-only and runs on a laptop.

The reference environment was Python 3.11.9 with PyTorch 2.x on an RTX-class GPU.
`torch.backends.cudnn.benchmark` is left enabled, matching the original notebooks, so
run-to-run variation is expected and is exactly what the five-run protocol measures.

---

## 4. Stage 1 — building the dataset

Notebook: `01_image_preparation_for_ML/image_preparation_pipeline.ipynb`.
All paths and parameters live in the first cell; nothing else needs editing.

Inputs per wall: an orthophoto (`*.png`, RGBA), a co-registered DEM (`*.tif`, same extent
and resolution), and one COCO JSON with the polygon annotations for the whole set.

The parameters that define the published dataset:

| Parameter | Value | Note |
|---|---|---|
| `CROP_SIZE` | 1280 | tile edge in pixels |
| `DESIRED_COVERAGE` | 1.6 | Sobol coverage factor |
| `NZ_MODE` / `NZ_EPS` | `fixed_eps` / 0.001 | out-of-plane component of the normal ("epsV1") |
| `APPLY_TILT_CORRECTION` | `True` | RANSAC yaw normalisation in normal space |
| `EXCLUDE_FROM_TRAINING` | wall1 … wall4 | held out before tiling |
| `BLACK_THRESHOLD` | 1.0 | drop fully empty mask tiles |

Notes that matter for reproduction:

- **Normal maps** follow the Mikkelsen convention, `n = (−dz/dx, +dz/dy, nz)` normalised.
  `nz` is the fixed 0.001 of the v1 pipeline; a per-wall `pixel_size` alternative is
  implemented (`NZ_MODE`) and was tested, but the fixed value is the published setting.
- **Yaw normalisation** fits the dominant wall normal by RANSAC with local refinement and
  rotates all normals so that it maps to (0, 0, 1) (Rodrigues). The rotation is isometric —
  local stone relief is preserved — and it is applied identically to training and test walls.
- **The test walls are separated automatically** in step 5.5, after normal-map generation and
  before tiling, so they can never enter the training tiles. Earlier versions of this README
  described a manual move; that instruction is obsolete.
- **Flipping happens before normal-map generation**, so the horizontal component of the
  normals is consistent with the flipped geometry. This is also why no mirror-based
  test-time augmentation is used at inference.

Output of a complete run: 2,149 tiles in each of `snippets_orthomosaics/`,
`snippets_normalmaps/` and `snippets_masks/`, plus the four held-out walls at full
resolution under `Testing/01_test-images`, `02_test-normals`, `03_test-masks`.

---

## 5. Stage 2 — training, segmentation and evaluation

Notebook: `05_orchestration/orchestrator.ipynb`. One config cell, then a preview cell that
checks every input path and prints every output path *before* anything heavy runs.

```python
BASE_CONFIG = RunConfig(
    channels=7, run_number=1,
    experiment_name="v2_yaw_correction-epsV1",
    experiments_root=".../experiments",
    ortho_dir=..., normalmap_dir=..., mask_dir=...,
    test_ortho_dir=..., test_normalmap_dir=..., test_mask_dir=...,
    seed=42, n_epochs=300, batch_size=16, lr=1e-4,
    roi_operation="closing", kernel_radius=45,
)
amg.run_sweep(BASE_CONFIG, channel_variants=(3, 4, 7), n_runs=5)
```

What one run does:

1. **train** — MultiUNet (2.16 M parameters), encoder widths [16, 32, 64, 128, 256], 512 px
   inputs, Adam(1e-4), 0.5 × CE + 0.5 × Dice with equal class weights, gradient clipping at
   1.0, 300 epochs, fixed 90/10 train/validation split. Saves `model.pth` (final epoch — the
   paper's convention), `model_best.pth` (best validation IoU, instrumentation only), and
   `config.json` with the full provenance of the run.
2. **segment** — sliding window 1280 px, stride 960, per-window softmax, center-weighted
   merge, written as a colour-coded raster `<wall>_RAW_combined.png`. A lightly cleaned
   grayscale raster is written alongside for visual inspection only; **metrics always use the
   RAW raster.**
3. **evaluate** — MC-ROI metrics: the region of interest is the morphological closing of the
   GT stone mask (disk, r = 45 px), which excludes out-of-bond detections. Per-class IoU and
   precision / recall / F1 within the ROI, written per wall and summarised per run, then
   appended to `experiments/<name>/manifest.csv`.

Before any sweep, run the equivalence check in section 2 of the notebook
(`amg.verify_architecture()`): it asserts exact parameter counts per variant and, if you point
it at an existing checkpoint, that the state dict loads with no missing or unexpected keys.

**Runtime**, RTX-class GPU: roughly 2–4 h per training run, so about 45–60 h for a full
(3, 4, 7) × 5 sweep. Segmentation of the four walls takes a few minutes per run; ROI
evaluation is CPU-bound and takes a few minutes per run.

---

## 6. Stage 3 — the ensemble that produced the published table

The paper does not report a single training run. For each variant the softmax outputs of the
five runs are averaged per window, and the averaged probabilities go through the unchanged
merge, argmax and evaluation path. The same procedure is applied to all three variants, so the
comparison stays fair; it costs no additional training.

```python
amg.run_ensemble(BASE_CONFIG,
                 source_experiment="v2_yaw_correction-epsV1",
                 out_experiment="v2_baseline_ens",
                 channel_variants=(3, 4, 7), n_runs=5)
```

Result: **`experiments/v2_baseline_ens/manifest.csv`** — the canonical source of the paper's
headline figures.

```
AllWalls, mean stone IoU     7ch 0.583 | 4ch 0.560 | 3ch 0.530
```

### Reproducing this without training

The checkpoints are deposited (§9), so the published table can be rebuilt without the ~50 GPU
hours of a full sweep. Unzip `AmG_checkpoints_v2_yaw_correction-epsV1.zip` into
`experiments/v2_yaw_correction-epsV1/checkpoints/`, so that each run lands at
`checkpoints/<run_id>/model.pth`, point the config cell at the unzipped dataset, and run the
`run_ensemble` call above. It loads the checkpoints, writes the rasters, evaluates them and
appends the manifest — a few minutes per variant on a CUDA GPU, considerably longer on CPU.

`05_orchestration/final_numbers.ipynb` runs this stage together with the other candidates that
were evaluated in parallel (per-run baseline, cosine-LR sweep, best-validation checkpoints,
cosine ensemble) and prints them side by side. The baseline ensemble is the one that was
adopted; the alternatives are documented in `experiments/` as recorded negative results and
are not used for any published number.

---

## 7. Stage 4 — the evaluations beyond pixel IoU

| What | Driver | Output |
|---|---|---|
| Gap detection and stone separation | `05_orchestration/gap_metrics.ipynb` | `experiments/v9_gapstone/gap_stone_metrics.csv` |
| Boundary IoU (Cheng et al. 2021), d = 5 and 15 px | `05_orchestration/stress_boundary_stage0.ipynb` | `experiments/v6_boundary/boundary_manifest.csv` |
| Lighting robustness (brightness, contrast, gamma, shadow) | same notebook | `experiments/v6_stress_<condition>/manifest.csv` |
| Training curves | `05_orchestration/training_curves.ipynb` | `experiments/v7_histories/training_histories.csv` |
| Ensemble confidence | `amg_pipeline/confidence.py`, driven from `orchestrator.ipynb` | `experiments/v2_baseline_ens/confidence/7ch_run1/` |

Two of these deserve a word, because they carry claims that pixel IoU cannot support:

**Gap and stone metrics.** Inter-stone gaps are a small fraction of the wall, so a model that
merges neighbouring stones can still score well on pixel IoU. Gaps are extracted as
`closing(mask, disk(45)) AND NOT mask`, on the GT and on the prediction, and compared inside
the ROI. Independently, each of the 663 GT stones (≥ 100 px) is scored for *coverage* (how much
of it is covered by same-class prediction) and *separation* (how committed the overlapping
predicted components are to this one stone). A stone counts as detected at 0.9 / 0.9, as merged
if it is covered but not separated. The thresholds were fixed before the results were seen; the
looser 0.7 / 0.7 and 0.5 / 0.5 pairs are reported in the same CSV.

**Confidence.** The ensemble inference is re-run keeping the merged per-pixel probabilities.
It writes a 16-bit confidence raster (top-1 probability), the full probability stack for one
wall, and a summary CSV. The column `raster_agreement` is a self-check: it must be ≈ 1.0, i.e.
the recomputed argmax reproduces the saved ensemble raster. If it is not, the pass did not
reproduce the ensemble and its statistics must not be used. The two figure scripts in
`04_segmentation_evaluation/` turn these outputs into the published panels.

---

## 8. Reproducing a specific number from the paper

Start from the CSV, not from a rerun. Every figure in the paper has a file behind it:

| Paper item | File | Regenerate with |
|---|---|---|
| Per-class and mean IoU / F1 per wall (main results table) | `experiments/v2_baseline_ens/manifest.csv` | `final_numbers.ipynb`, stage C |
| Run-to-run spread, mean ± std | `experiments/v2_yaw_correction-epsV1/manifest.csv` and `metrics/<run_id>/roi_summary_*_closing.csv` | `orchestrator.ipynb`, section 5 |
| Gap IoU, detection and merge rates | `experiments/v9_gapstone/gap_stone_metrics.csv` | `gap_metrics.ipynb` |
| Boundary IoU | `experiments/v6_boundary/boundary_manifest.csv` | `stress_boundary_stage0.ipynb` |
| Lighting robustness | `experiments/v6_stress_*/manifest.csv` | `stress_boundary_stage0.ipynb` |
| Confidence statistics | `experiments/v2_baseline_ens/confidence/7ch_run1/confidence_summary.csv` | `orchestrator.ipynb`, confidence cell |
| Training-set size ablation (25 / 50 / 75 %) | `experiments/v5_fractioning_frac*/manifest.csv` | `orchestrator.ipynb` with `train_fraction` |
| Training curves | `experiments/v7_histories/training_histories.csv` | `training_curves.ipynb` |

Full experiment index, including the variants that were tested and rejected:

| Experiment | What it screened | Status |
|---|---|---|
| `v2_yaw_correction-epsV1` | published recipe, 3/4/7 × 5 runs | **source of the published ensemble** |
| `v2_baseline_ens` | mean-softmax ensemble of the above | **published figures** |
| `v2_NO_yaw_correction`, `v2_yaw_corrected` | normal-map attribution study (tilt on/off, eps variants) | settled: fixed eps, tilt retained |
| `v3_architecture-size_slim`, `_wide` | encoder width | negative |
| `v4_oversample_w3`, `v4_dice070_no-w` | class imbalance: quarry oversampling, CE/Dice ratio | negative |
| `v5_fractioning_frac25/50/75` | training-set size | reported ablation |
| `v6_boundary` | Boundary IoU on the frozen rasters | reported |
| `v6_stress_*` | lighting robustness, eval-only | reported |
| `v7_groupnorm`, `v7_photoaug` | GroupNorm, photometric augmentation | negative |
| `v8_coslr`, `v8_coslr_best`, `v8_coslr_ens` | cosine LR schedule, best-val checkpoints | negative, not used |
| `v9_gapstone` | gap and stone-separation metrics | reported |

The `v6_stress_none` condition re-runs the unperturbed inputs through the whole stress harness
and must reproduce the frozen baseline manifest — a built-in check on that harness.

---

## 9. Data availability

Everything needed to reproduce the published results, apart from this code, is in one Zenodo
record:

**[10.5281/zenodo.21507975](https://doi.org/10.5281/zenodo.21507975)** (CC-BY 4.0)

| File | Contents |
|---|---|
| `2025_AmG_TrainingTestingDataCompress.zip` (4.9 GB) | orthophotos, DEMs, COCO annotations, generated masks and normal maps, the Sobol tiles, and the four held-out test walls at full resolution |
| `AmG_checkpoints_v2_yaw_correction-epsV1.zip` | the 15 trained checkpoints behind the published ensemble — `{3,4,7}ch_run{1..5}`, each with its `config.json`, plus SHA256 checksums |

Unzipped, the dataset's folder layout is the one the notebooks expect, so the path blocks in the
config cells can be pointed at it directly with no restructuring. `model.pth` is the final-epoch
checkpoint and is the published model; `model_best.pth` is not included and was not used for any
published number.

A citable release of this repository is archived separately.
<!-- TODO: insert the code-release DOI once the v2.0 GitHub release is minted -->

Full-resolution prediction rasters are not deposited. They are regenerated from the checkpoints
in minutes (§6), and the metrics computed from them are in `experiments/`.

---

## 10. Conventions and pitfalls

Anyone rebuilding these results should read this section first — most of it is invisible in
the code and each item can shift a number.

**Classes.**

| Class | Index | Colour | Raw mask value |
|---|---|---|---|
| Background | 0 | black `#000000` | 0 |
| Ashlar | 1 | blue `#0000FF` | 29 |
| Polygonal | 2 | red `#FF0000` | 76 |
| Quarry stone | 3 | yellow `#FFFF00` | 225 |

**Absent classes are dropped, not scored zero.** A class with no GT pixels inside a wall's ROI
is recorded as NaN and excluded from that wall's mean; it is not counted as 0.0. Averaging
zeros over absent classes would deflate every mean and would penalise variants inconsistently.
In tables, absent classes are marked "–". The AllWalls aggregate averages each class only over
the walls where it is present.

**Class-first averaging.** The AllWalls macro figures average per class across walls first, and
only then across classes — matching the IoU path. Averaging wall-first gives a different macro
F1. This is a convention, not a correction; it is applied consistently everywhere.

**Channel order in the 7-channel stack.** Orthophotos are read with `cv2.imread(...,
IMREAD_UNCHANGED)` and normal maps with `IMREAD_COLOR`, both of which return BGR(A), while the
preparation pipeline writes normal-map PNGs as RGB in the order (nx, ny, nz). The stack the
network actually sees is therefore `[B, G, R, A, nz, ny, nx]`. This is consistent between
training and inference and is not a bug, but any code that builds the stack differently — for
instance reading the normal map back from a GeoTIFF — must reproduce this order explicitly.

**RAW versus cleaned rasters.** `<wall>_RAW_combined.png` is the metric input.
`<wall>_segmented.png` has had a median blur and a morphological opening applied and exists for
visual inspection only. Never evaluate the cleaned raster.

**Final-epoch checkpoints.** `model.pth` is the final epoch and is what the paper uses.
`model_best.pth` (best validation IoU) is written alongside as instrumentation and was
evaluated as a separate candidate; it is not the published model.

**Never report a best single run.** Either the five-run ensemble or mean ± std across the five
runs. The manifests contain every individual run so this stays checkable.

**Wall 4 is the counter-example.** The fused variant does not lead on every wall. Any summary
that claims it does is wrong, and the per-wall rows in the manifest show why.

---

## 11. Citation

```bibtex
@inproceedings{schnorr2025appearance,
  title     = {Appearance Meets Geometry: Deep Learning for Semantic Segmentation
               of Archaeological Fortification Masonry},
  author    = {Schnorr, Nils and Leimk{\"u}hler, Thomas},
  booktitle = {Proceedings of the International Conference on Computer Applications
               in Archaeology (CAA)},
  year      = {2025},
  address   = {Athens, Greece},
  doi       = {10.5281/zenodo.21507975}
}
```

## 12. License

MIT — see [LICENSE](LICENSE).

---

**Repository:** https://github.com/NilsSchnorr/AppearanceMeetsGeometry
**Last updated:** July 2026
