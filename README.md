# ⛵ SailNet: Multi-Task Frame Classification

**Frame-level classification of point of sail, tack and boat type from sailing video.**

University project by **Nicolò Nucci** and **Chiara Mezzanzanica**
Università degli Studi di Milano-Bicocca

---

## Overview

Reviewing sailing footage is slow, and the annotations that come out of it are still made by hand. **SailNet** is a multi-task convolutional classifier that labels every frame of a sailing video with three attributes at the same time:

| Output | Classes | Notes |
|---|---|---|
| **Point of sail** (direction) | no-go zone, close-hauled, close reach, beam reach, broad reach, run | Only 3 of the 6 classes are present in our footage: no-go zone, close-hauled and run |
| **Tack** (side) | port, starboard | Mainly a spatial problem: the sail sits on opposite sides of the hull |
| **Boat type** | ILCA, Optimist | |

All models share an **EfficientNetV2-B2** backbone (ImageNet weights) and a common neck, followed by three prediction heads. The main design question is *how the heads should exchange information*, so we compare four architectures under identical data and augmentation:

| Model | Description |
|---|---|
| **A** | Independent heads |
| **B** | Full cascade (boat → side → direction) |
| **C** | Partial cascade (boat → direction only) |
| **D** | Partial cascade + lightweight spatial attention (CBAM-lite) on the tack head |

Training uses a focal loss (γ = 2) with label smoothing, on-the-fly [Albumentations](https://albumentations.ai/) augmentation (deterministic seeds, no horizontal flip since it would swap port and starboard), and a two-phase protocol: a frozen-backbone phase followed by progressive partial unfreezing.

## Key findings

- **Cascade connections gave no benefit.** Passing boat-type (or tack) logits to the other heads did not help.
- **Spatial attention on the tack head helped markedly.** On validation, tack accuracy rose from 0.69 (Model C, same cascade) to 0.82 (Model D); partial fine-tuning raised it to 0.87.
- **Tack generalises to unseen sessions.** On the held-out test set (sessions never seen during training), Model D reaches a macro accuracy of **0.85** for tack, a solid base for detecting tacks and gybes.
- **Boat type and point of sail are less mature.** Boat type generalises worse than validation suggested (macro accuracy 0.88 on test), and the no-go zone is not yet recognised: the class is very scarce in our data (13 training sessions, about 22 seconds of real video).

| Test set (Model D, after Phase 2) | Macro accuracy |
|---|---|
| Tack | 0.852 |
| Boat type | 0.884 |
| Point of sail | 0.594 |

> These results come from a single training run with one seed and a small test split (about 15% of 95 sessions). See the report for a full discussion of limitations.

## Dataset

The dataset was built from **95 on-water training sessions** (8,775 frames) using a reproducible extraction and labelling pipeline:

- frames extracted with `ffmpeg` (OpenCV as fallback) at 5 fps (15 fps for the no-go zone), resized to 224 × 224;
- one label triple (point of sail, tack, boat type) per video;
- **session-level** stratified train / validation / test split (65 / 20 / 15) so that near-identical neighbouring frames never leak between partitions.

### ⚠️ Data is not included

**The dataset (raw videos and extracted frames) is not published in this repository.** The footage shows youth sailing training sessions in which minors are clearly identifiable, so it cannot be shared for privacy reasons. The labelling notebook documents the exact pipeline and the expected input format, so the dataset can be rebuilt from your own footage.

## Repository structure

```
.
├── README.md
├── SailNet.pdf          # Report (methodology, experiments, discussion)
├── SailNet_Presentation.pptx   # Project slides
└── notebooks/
    ├── Sailing_Labeling.ipynb      # Video → frames → labelled manifests (train/val/test)
    └── Sailing_Training.ipynb      # Model definition, training and evaluation
```

## Notebooks

Both notebooks were developed and run on **Google Colab** (they mount Google Drive for data and checkpoints). Run them in order:

1. **`Sailing_Labeling.ipynb`** reads a spreadsheet (`video_jobs.xlsx`) with one row per video (`video`, `direction`, `side`, `boat_type`), checks that labels are valid and files exist, extracts frames, performs the session-level split, and produces `train.csv`, `val.csv`, `test.csv`, `class_info.json` and distribution plots.
2. **`Sailing_Training.ipynb`** reads the manifests and the class information file produced above, trains the multi-head models with a custom Keras `Sequence` and Albumentations augmentation, and evaluates them on the held-out test set.

### Requirements

Main libraries: `tensorflow` / `keras`, `albumentations`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`, `tqdm`, plus `ffmpeg` for frame extraction.

To run outside Colab, replace the Google Drive mount and paths at the top of each notebook with local ones.

## Future work

- **Data:** footage for close reach, beam reach and broad reach, and more varied no-go zone footage (different boats, venues, conditions); repeated runs with several seeds and session-level cross-validation.
- **Temporal modelling:** a sequence model (e.g. bidirectional LSTM or Transformer) over the per-frame outputs to detect tacks and gybes and extract session-level statistics.
- **More boat classes** (e.g. 470, 49er) with new labelled data and no change to the architecture.

## Citation / References

Key references used in this work: EfficientNetV2 (Tan & Le, ICML 2021), Focal Loss (Lin et al., ICCV 2017), CBAM (Woo et al., ECCV 2018), Albumentations (Buslaev et al., 2020). The full list is in the report.

## Authors

- **Nicolò Nucci** — n.nucci@campus.unimib.it
- **Chiara Mezzanzanica** — c.mezzanzanica2@campus.unimib.it

Università degli Studi di Milano-Bicocca

## Use of AI tools

Large language models were used as writing tools during this project to help write and debug the notebook code. Research questions, experimental design, architecture and hyper-parameter choices, interpretation of results and conclusions are the authors' own.
