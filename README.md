# Cerebral microbleed detection on T2* with a 3D Attention U-Net (VALDO Task 2)

Counts cerebral microbleeds (CMB) on T2*-weighted MRI. The input is T2* plus an
FRST channel, the network is a 3D Attention U-Net, and predictions are
post-processed via connected components (26-connectivity) and a T1-based filter
that removes false positives.

**Status: the pipeline is set up, results are pending.** The tables below are
filled from the runs and are empty until then.

## Why counting, not Dice

A microbleed is a few voxels across. A Dice score over a whole volume says
almost nothing about it — a network can miss half the bleeds and still post a
respectable Dice. So the metric operates per bleed instead.

Prediction and reference are each split into connected components with
26-connectivity (`ndimage.label(m, structure=np.ones((3,3,3)))`); one component
is one bleed. Then:

- **TP** — a reference component touched by at least one predicted component
- **FN** — a reference component that is not touched
- **FP** — a predicted component that touches no reference component
- several predictions on **one** reference count as 1 TP and 0 FP
- **F1** = 2TP / (2TP + FP + FN), pooled across all cases

Reported alongside: false positives per case and the counting error
|n_predicted − n_reference|. True negatives exist only at case level.
Implementation: `code/metric.py` (run `--selftest`).

## Dataset

VALDO Challenge Task 2 (MICCAI 2021): 72 cases, 236 CMB. Three cohorts with very
different acquisition geometry — for a 3D network this is the decisive property,
so the pipeline addresses it explicitly:

| Cohort | n | CMB | voxel size (mm) | anisotropy |
|---|---|---|---|---|
| 1 | 11 | 106 | 0.45 × 0.45 × 4.0 | 8.9:1 |
| 2 | 34 | 96 | 0.49 × 0.49 × 0.8 | 1.6:1 |
| 3 | 27 | 34 | 1.0 × 1.0 × 3.0 / 4.0 | 3–4:1 |

A fixed patch of 96×64×48 spans 192 mm in z for cohort 1 but only 38 mm for
cohort 2. Whether a common grid (0.5 × 0.5 × 1.0 mm) beats native-resolution
training is measured, not assumed.

T1, T2, T2* and the CMB mask share one grid in 72/72 cases (shape and affine
verified), and the images are already skull-stripped. Two consequences: no brain
extraction is needed here, and the T1-based filter needs **no registration**.

**The data is not included** — CC-BY-NC-4.0 does not permit redistribution — and
is obtained directly from the VALDO challenge.

## Pipeline

1. **Preprocessing** — FSL FAST `-B` (bias correction), FRST channel and vessel
   overpainting following the microbleednet procedure.
2. **Split** — ~20 % of cases held out for validation, stratified by cohort and
   lesion load (0 / 1–2 / ≥3) with a greedy balance of the CMB totals. The
   remainder is 5-fold cross-validated.
3. **Hyperparameter search** — Optuna (TPE) with pruning on a single fold; the
   three best configurations are then confirmed across all five folds.
4. **Post-processing** — a minimum size in mm³ (not in voxels: the cohorts differ
   greatly in voxel volume), then a filter discarding candidates that fall in
   ventricles or CSF (SynthSeg on the supplied T1).

**The pooled cross-validation numbers are the headline result.** The held-out set
contains roughly 47 bleeds; an F1 read from it carries about ±0.1 of uncertainty
and serves as a sanity check, not as the result. Thresholds, minimum sizes and
hyperparameters are derived from training folds only.

## Results

Pooled 5-fold cross-validation (training portion):

| Method | F1 | Precision | Sensitivity | FP per case |
|---|---|---|---|---|
| (pending) | | | | |

Held-out validation set, measured once:

| Method | F1 | Precision | Sensitivity | FP per case |
|---|---|---|---|---|
| (pending) | | | | |

## Sources

Which step comes from where:

| Step | Source |
|---|---|
| Dataset | VALDO Challenge Task 2, MICCAI 2021 — https://valdo.grand-challenge.org (CC-BY-NC-4.0) |
| FRST channel, vessel overpainting (`inpaint_vessels`), candidate detection + discrimination | Sundaresan et al., *microbleednet* — https://github.com/v-sundaresan/microbleednet |
| Fast radial symmetry transform (basis of the FRST channel) | Loy & Zelinsky, *Fast Radial Symmetry for Detecting Points of Interest*, IEEE TPAMI 2003 |
| Network architecture | Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, 2018 |
| Architecture implementation, patch sampling, augmentation | MONAI — https://monai.io |
| Bias correction (`fast -B`) | Zhang, Brady & Smith, *Segmentation of brain MR images through a hidden Markov random field model*, IEEE TMI 2001 (FSL FAST) |
| T1 segmentation for the false-positive filter | Billot et al., *SynthSeg*, Medical Image Analysis 2023 (FreeSurfer `mri_synthseg`) |
| Hyperparameter search (TPE, pruning) | Akiba et al., *Optuna*, KDD 2019 |
| Brain extraction | Isensee et al., *HD-BET*, Human Brain Mapping 2019 — **not used in this pipeline**, since the VALDO images are already skull-stripped; kept in the code for cohorts that are not |

## Environment

Python 3, PyTorch 2.6 (CUDA 12.4), MONAI 1.5.2, Optuna 5.0, nibabel, scipy,
FSL 6.x, FreeSurfer 8.2.
