# Contour-Based Digit Recognition via Fourier Descriptors

This project implements a contour-based OCR pipeline for Persian digit recognition using  
**Fourier Descriptors (FD)** and a **k-Nearest Neighbors (kNN)** classifier.  
The approach is fully shape-based, lightweight, and requires no deep learning.

---

## Overview

The system extracts external contours from a page of handwritten Persian digits and converts each contour into a translation-, scale-, rotation-, and start-invariant Fourier Descriptor.  
Classification is performed using a single-stage, distance-weighted **kNN (k=5)** model.

According to the project report (MiniProject-6), the pipeline achieves **0.963 accuracy** for 20%, 10%, 5%, and 2% FD retention, with a slight decrease to **0.950** for 1% retention.  
These results are obtained under a **Leave-One-Row-Out (LORO)** evaluation protocol. :contentReference[oaicite:1]{index=1}

---

## Key Features

- Robust preprocessing: Otsu thresholding, external contour extraction, row grouping  
- Cleanup: noise filtering, bullet removal, merged-component splitting  
- Uniform contour resampling to **512 points**  
- Translation / scale / rotation invariance via:
  - Zeroing DC term
  - Normalizing by `|C1|`
  - Phase-locking to `C1`
- FD truncation at multiple retention levels: **20%, 10%, 5%, 2%, 1%**
- Lightweight classifier: **kNN (k=5, distance-weighted)**
- Full evaluation: accuracy, confusion matrix, LORO visual overlays
- Automatic export of all plots and metrics

---

## Pipeline

The complete algorithm follows the stages described in the report (pages 1–3): :contentReference[oaicite:2]{index=2}

1. **Preprocessing**
   - Grayscale → Otsu threshold (inverted)
   - Extract external contours  
   - Group components into 8 rows (adaptive gaps; fallback K-Means)

2. **Row Cleanup**
   - Remove rightmost bullet (if present)  
   - Filter small/noisy components  
   - Split merged blobs  
   - Keep the leftmost 10 components and assign labels 9→0

3. **Fourier Descriptor Extraction**
   - Arc-length resampling (N = 512)
   - Complex contour representation
   - Invariance steps: centering, scaling, phase-locking
   - Feature vector = `[Re(C1..M), Im(C1..M)]`

4. **Classification**
   - kNN classifier (k=5, distance-weighted)

5. **Evaluation**
   - Leave-One-Row-Out protocol
   - Confusion matrices + visual labeled overlays

---

## Project Structure

## Project Structure

```text
.
├── main.py
├── Im321.png
├── outputs/
│   ├── plots/
│   ├── metrics/
│   └── config.txt
└── MV6_Sadeghi6.pdf
```
## Installation
pip install opencv-python numpy matplotlib scikit-learn

## Running the Project
python main.py

Outputs will appear under the outputs/ directory, including:
LORO labeled predictions
Confusion matrix heatmaps
Accuracy and per-class metrics
Run configuration

## Documentation
Full algorithm explanation, parameter tables, and experimental results are available in:
MV6_Sadeghi6.pdf
