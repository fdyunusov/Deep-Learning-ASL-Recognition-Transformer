# ASL Transformer — WLASL-100 Pipeline

Transformer-based American Sign Language recognition trained from scratch on skeleton keypoint sequences extracted from the WLASL-100 video dataset. No CNN, no RNN.

---

## Project Structure

```
asl/
├── WLASL_v0.3.json              # Full WLASL dataset index (2000 classes)
├── WLASL100.json                # Filtered index (100 classes) — generated
├── wlasl100_lookup.json         # video_id → label/split map — generated
├── wlasl100_stats.txt           # Split statistics — generated
├── keypoints.npz                # Extracted keypoints — generated
├── mlp_baseline_best.pt         # Best MLP checkpoint — generated
│
├── video_downloader.py          # Step 1: Download raw videos from WLASL URLs
├── preprocess.py                # Step 2: Extract per-instance video clips
├── filter_wlasl100.py           # Step 3: Filter to WLASL-100 (100 classes)
├── extract_keypoints.py         # Step 4: Run MediaPipe, save keypoints.npz
├── dataloader_and_baseline.py   # Step 5: DataLoader + MLP baseline training
│
├── raw_videos/                  # Downloaded raw videos (non-YouTube)
├── raw_videos_mp4/              # Converted mp4 versions
└── videos/                      # Per-instance clips — output of preprocess.py
```

---

## Setup

**Requirements**

```
pip install yt-dlp mediapipe==0.10.9 opencv-python numpy torch scikit-learn tqdm
```

**Note on yt-dlp:** if `yt-dlp` is not recognized after installing, use:
```
python -m yt_dlp --version
```
and change line 13 of `video_downloader.py` to:
```python
youtube_downloader = "python -m yt_dlp"
```

---

## Pipeline — Run in Order

### Step 1 — Download raw videos
```
python video_downloader.py
```
Downloads all non-YouTube and YouTube source videos from the WLASL URL list into `raw_videos/`. YouTube downloads use yt-dlp. Expect 30–60 minutes for 100 classes. Some URLs will fail (dead links) — this is normal.

### Step 2 — Extract per-instance clips
```
python preprocess.py
```
Converts `.swf` and `.mkv` files to `.mp4`, then extracts the exact frame ranges specified in the JSON for each instance into `videos/`. Output is one `.mp4` file per video instance.

### Step 3 — Filter to WLASL-100
```
python filter_wlasl100.py
```
Slices the full 2000-class JSON to the top 100 glosses. Outputs `WLASL100.json` and `wlasl100_lookup.json` (flat video_id → gloss/label/split map).

Expected output:
```
Classes : 100
Train   : 1442 videos
Val     :  338 videos
Test    :  258 videos
Total   : 2038 videos
```

### Step 4 — Extract keypoints
```
python extract_keypoints.py
```
Runs MediaPipe Holistic on every video in `videos/` that belongs to WLASL-100. Extracts per-frame skeleton keypoints:

| Landmark group | Landmarks | Values per frame |
|---|---|---|
| Pose | 33 × (x, y, z) | 99 |
| Left hand | 21 × (x, y, z) | 63 |
| Right hand | 21 × (x, y, z) | 63 |
| **Total** | | **225** |

Sequences are padded or truncated to 64 frames. Saves `keypoints.npz`:
```
X       → float32  (N, 64, 225)   keypoint sequences
y       → int64    (N,)           label indices
splits  → str      (N,)           'train' / 'val' / 'test'
ids     → str      (N,)           video_id strings
glosses → str      (100,)         class names in label order
```

### Step 5 — DataLoader + MLP Baseline
```
python dataloader_and_baseline.py
```
Loads `keypoints.npz`, builds PyTorch DataLoaders with z-score normalization and augmentation (Gaussian noise + random frame dropout on train set), then trains the MLP baseline for 100 epochs.

Expected output:
```
── Test Set Results (MLP Baseline) ──
  Top-1 Accuracy : ~15–30%
  Top-5 Accuracy : ~40–60%
  Macro F1-Score : ~14–28%
  Random chance  :   1.00%
```
Accuracy well above 1% confirms the full pipeline is working correctly.

---

## Dataset

**WLASL (Word-Level American Sign Language)**
- Homepage: https://dxli94.github.io/WLASL/
- This project uses the **WLASL-100** subset (top 100 most frequent glosses)
- WLASL is a fully video-based dataset — raw RGB video clips of signers, recorded across multiple signers and real-world environments
- Keypoint extraction via MediaPipe Holistic is applied as a preprocessing step; the model never sees raw pixels

---

## Models

| Model | Who | Status |
|---|---|---|
| MLP Baseline | Shared | Step 5 above |
| Vanilla Keypoint Transformer (VKT) | Member 1 (Fuad) | Week 2 |
| Siformer | Member 2 | Week 2 |

---

## Evaluation Metrics

| Metric | Target |
|---|---|
| Top-1 Accuracy | >50% (VKT) |
| Top-5 Accuracy | >75% (VKT) |
| Macro F1-Score | >0.48 (VKT) |
| Confusion Matrix | Qualitative analysis |

---

## References

- Li et al. (2020). WLASL: A Large-Scale Dataset for Word-Level ASL. https://arxiv.org/abs/1910.11006
- Gong et al. (2024). Siformer. ACM MM. https://doi.org/10.1145/3664647.3681578
- Berepiki et al. (2025). Lightweight Transformer for ASL. ICCV Workshop.
- Bohacek & Hruz (2022). Sign Pose-Based Transformer. WACV Workshop.
- Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
