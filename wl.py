"""
Step 3: Filter WLASL to the top 100 glosses (WLASL-100 split).

Usage:
    python filter_wlasl100.py

Input:  WLASL_v0.3.json  (full 2000-class dataset)
Output: WLASL100.json    (100 classes, 2038 instances total)
        wlasl100_stats.txt

Place this script in the same folder as WLASL_v0.3.json.
"""

import json
import os

INPUT_JSON  = "WLASL_v0.3.json"
OUTPUT_JSON = "WLASL100.json"
NUM_CLASSES = 100

# ── Load and slice ────────────────────────────────────────────────────────────
data = json.load(open(INPUT_JSON))
wlasl100 = data[:NUM_CLASSES]

# ── Save filtered JSON ────────────────────────────────────────────────────────
with open(OUTPUT_JSON, "w") as f:
    json.dump(wlasl100, f, indent=2)

# ── Print stats ───────────────────────────────────────────────────────────────
glosses = [entry["gloss"] for entry in wlasl100]
train_ids, val_ids, test_ids = [], [], []

for entry in wlasl100:
    for inst in entry["instances"]:
        vid = inst["video_id"]
        if inst["split"] == "train":
            train_ids.append(vid)
        elif inst["split"] == "val":
            val_ids.append(vid)
        elif inst["split"] == "test":
            test_ids.append(vid)

total = len(train_ids) + len(val_ids) + len(test_ids)

stats = f"""
WLASL-100 Statistics
====================
Classes : {NUM_CLASSES}
Glosses : {', '.join(glosses[:10])} ... {glosses[-1]}

Split breakdown:
  Train : {len(train_ids)} videos
  Val   : {len(val_ids)} videos
  Test  : {len(test_ids)} videos
  Total : {total} videos

Saved to: {OUTPUT_JSON}
"""
print(stats)

with open("wlasl100_stats.txt", "w") as f:
    f.write(stats)

# ── Build a flat lookup: video_id -> {gloss, label_idx, split} ───────────────
lookup = {}
for label_idx, entry in enumerate(wlasl100):
    for inst in entry["instances"]:
        lookup[inst["video_id"]] = {
            "gloss":     entry["gloss"],
            "label_idx": label_idx,
            "split":     inst["split"],
        }

with open("wlasl100_lookup.json", "w") as f:
    json.dump(lookup, f, indent=2)

print(f"Lookup table saved to wlasl100_lookup.json ({len(lookup)} entries).")