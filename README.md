# Learnable MSFA Extension Repository

This repository contains the extension track for multispectral filter array (MSFA) research built around OSP priors and controlled branch comparisons.

It is organized for clear reproduction of the experimental pipeline, figure generation, and manuscript preparation.

## 1) Repository Goal

The core goal is a fair and interpretable comparison of three sensing branches under matched conditions:

1. Fixed OSP baseline
2. Exact OSP selector (discrete candidate selection)
3. OSP-seeded learnable MSFA

All three branches are evaluated with the same split policy, reconstruction family, and reporting protocol.

## 2) Repository Layout

- [notebooks](notebooks): phase-wise notebooks (Phase 1 to Phase 11)
- [figures](figures): exported figures used by the manuscript
- [paper](paper): manuscript source and compiled PDF
- [scripts](scripts): utility scripts
- [.gitignore](.gitignore): ignore rules for temporary files

### Folder Tree

```text
extension/
├─ notebooks/
│  ├─ 01_phase1_data_protocol.ipynb
│  ├─ 02_phase2_fixed_msfa_baseline.ipynb
│  ├─ 03_phase3_learnable_msfa.ipynb
│  ├─ 04_phase4_sp_regularized_msfa.ipynb
│  ├─ 05_phase5_ablations_figures_tables.ipynb
│  ├─ 06_phase6_learned_method_figures.ipynb
│  ├─ 07_phase7_export_to_matlab.ipynb
│  ├─ 08_phase8_3d_osp_regularized_msfa.ipynb
│  ├─ 09_phase9_learnable_ab_bridge.ipynb
│  ├─ 10_phase10_discrete_osp_selector.ipynb
│  └─ 11_phase11_osp_seeded_learnable_msfa.ipynb
├─ figures/
│  ├─ phase11_best_metric_bars.png
│  ├─ phase11_multiseed_errorbars.png
│  ├─ phase11_pattern_comparison.png
│  ├─ phase11_psnr_comparison.png
│  ├─ phase11_recon_and_error.png
│  ├─ phase11_sam_comparison.png
│  ├─ phase6_centroid_maps_phase11.png
│  └─ phase6_hard_tiles_phase11.png
├─ paper/
│  ├─ paper_draft_ieee.tex
│  └─ paper_draft_ieee.pdf
├─ scripts/
│  └─ generate_phase_notebooks.py
├─ .gitignore
└─ README.md
```

## 3) Phase-by-Phase Notebook Guide

### Phase 1 to Phase 4

1. [01_phase1_data_protocol.ipynb](notebooks/01_phase1_data_protocol.ipynb)
   - Dataset protocol and patch generation policy
2. [02_phase2_fixed_msfa_baseline.ipynb](notebooks/02_phase2_fixed_msfa_baseline.ipynb)
   - Fixed MSFA baseline branch
3. [03_phase3_learnable_msfa.ipynb](notebooks/03_phase3_learnable_msfa.ipynb)
   - Early free learnable branch
4. [04_phase4_sp_regularized_msfa.ipynb](notebooks/04_phase4_sp_regularized_msfa.ipynb)
   - SP-regularized learnable branch

### Phase 5 to Phase 7

5. [05_phase5_ablations_figures_tables.ipynb](notebooks/05_phase5_ablations_figures_tables.ipynb)
   - Comparison/ablation-oriented reporting notebook
6. [06_phase6_learned_method_figures.ipynb](notebooks/06_phase6_learned_method_figures.ipynb)
   - Figure production notebook
7. [07_phase7_export_to_matlab.ipynb](notebooks/07_phase7_export_to_matlab.ipynb)
   - MATLAB export utilities

### Phase 8 to Phase 11

8. [08_phase8_3d_osp_regularized_msfa.ipynb](notebooks/08_phase8_3d_osp_regularized_msfa.ipynb)
   - 3D OSP-regularized extension
9. [09_phase9_learnable_ab_bridge.ipynb](notebooks/09_phase9_learnable_ab_bridge.ipynb)
   - Learnable bridge toward OSP-style parameterization
10. [10_phase10_discrete_osp_selector.ipynb](notebooks/10_phase10_discrete_osp_selector.ipynb)
    - Exact discrete candidate selection branch
11. [11_phase11_osp_seeded_learnable_msfa.ipynb](notebooks/11_phase11_osp_seeded_learnable_msfa.ipynb)
    - Main unified notebook and primary result source

## 4) Experimental Protocol (Current Manuscript Configuration)

From [paper/paper_draft_ieee.tex](paper/paper_draft_ieee.tex):

1. Number of spectral bands: 16
2. Split: 24 train scenes, 8 validation scenes
3. Patch size: 128 x 128
4. Number of seeds: 3
5. Primary metrics: PSNR, SAM (deg)
6. Secondary metric: RGB-SSIM

Dataset note:
- Patches are extracted from CAVE scenes following the fixed split policy used by the manuscript.

## 5) Current Validated Main Results (Phase 11 Summary)

Branch-wise validation summary (mean +- std):

1. Fixed OSP: 34.463 +- 1.232 dB, 7.460 +- 0.889 deg, 0.8861 +- 0.0198 RGB-SSIM
2. Exact OSP: 34.756 +- 0.958 dB, 7.060 +- 0.486 deg, 0.8992 +- 0.0077 RGB-SSIM
3. Learnable: 38.142 +- 0.634 dB, 5.019 +- 0.333 deg, 0.9413 +- 0.0020 RGB-SSIM

Relative gain versus Fixed OSP:

1. Exact OSP: +0.293 PSNR, -0.400 SAM, +0.0131 RGB-SSIM
2. Learnable: +3.679 PSNR, -2.441 SAM, +0.0552 RGB-SSIM

## 6) Paper and Figure Mapping

Manuscript source:
- [paper/paper_draft_ieee.tex](paper/paper_draft_ieee.tex)

Compiled draft:
- [paper/paper_draft_ieee.pdf](paper/paper_draft_ieee.pdf)

Figures referenced by manuscript:

1. [figures/phase11_pattern_comparison.png](figures/phase11_pattern_comparison.png)
2. [figures/phase11_best_metric_bars.png](figures/phase11_best_metric_bars.png)
3. [figures/phase11_multiseed_errorbars.png](figures/phase11_multiseed_errorbars.png)
4. [figures/phase11_psnr_comparison.png](figures/phase11_psnr_comparison.png)
5. [figures/phase11_sam_comparison.png](figures/phase11_sam_comparison.png)
6. [figures/phase11_recon_and_error.png](figures/phase11_recon_and_error.png)
7. [figures/phase6_hard_tiles_phase11.png](figures/phase6_hard_tiles_phase11.png)
8. [figures/phase6_centroid_maps_phase11.png](figures/phase6_centroid_maps_phase11.png)

## 7) How To Build The Manuscript

From the repository root:

```powershell
Set-Location .\paper
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_draft_ieee.tex
```

The build writes PDF and temporary files inside [paper](paper).

Temporary LaTeX files are ignored by [.gitignore](.gitignore).

## 8) Recommended Reproduction Order

For a clean rerun:

1. Run Phase 1 once to set data protocol assumptions
2. Run baseline/comparison phases (2, 10, 11) for the main branch comparison
3. Run Phase 5 and Phase 6 if figures or reporting artifacts must be refreshed
4. Rebuild manuscript in [paper](paper)

## 9) Notes For Contributors

1. Keep branch reporting order consistent: Fixed OSP -> Exact OSP -> Learnable
2. Do not mix incompatible metrics (for example, RGB-SSIM vs other SSIM protocols) without explicit note
3. Keep manuscript claims aligned with reproducible notebook outputs
4. If figure files are renamed or moved, update figure references in [paper/paper_draft_ieee.tex](paper/paper_draft_ieee.tex)

## 10) License and Upstream Context

This extension sits alongside the original baseline project in the parent workspace.
Refer to upstream licensing and citation requirements when publishing derivative outputs.

## 11) Video Demo (Manim)

A ready-to-render Manim scene for a 15-20 minute technical walkthrough is provided in:

1. [scripts/manim_paper_demo.py](scripts/manim_paper_demo.py)

Execution and environment setup steps are documented in:

1. [scripts/MANIM_RUNBOOK.md](scripts/MANIM_RUNBOOK.md)
