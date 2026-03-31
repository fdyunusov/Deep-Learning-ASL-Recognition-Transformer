"""
Step 5: PyTorch DataLoader + MLP Baseline.

Loads keypoints.npz produced by extract_keypoints.py and provides:
  - ASLDataset        : PyTorch Dataset with normalization + augmentation
  - get_dataloaders() : returns train / val / test DataLoaders
  - MLPBaseline       : 3-layer MLP that operates on mean-pooled keypoints
  - train_baseline()  : trains the MLP and reports validation accuracy

Usage:
    pip install torch numpy scikit-learn
    python dataloader_and_baseline.py

Expected output: MLP validation accuracy well above 1% (random chance on 100 classes).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_FILE   = "keypoints.npz"
SEQ_LEN    = 64
N_FEATURES = 225     # 33*3 + 21*3 + 21*3 keypoints per frame
N_CLASSES  = 100
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3
NOISE_STD  = 0.01    # Gaussian noise augmentation std
FRAME_DROP = 0.1     # probability of zeroing a frame (augmentation)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────
class ASLDataset(Dataset):
    """
    Wraps the extracted keypoint sequences.

    X shape: (N, SEQ_LEN, N_FEATURES)  float32
    y shape: (N,)                       int64

    Normalization: per-feature z-score using training set statistics.
    Augmentation (train only):
      - Gaussian noise on keypoint coordinates
      - Random frame dropout (zero out a random subset of frames)
    """

    def __init__(self, X, y, mean=None, std=None, augment=False):
        """
        X      : np.ndarray (N, T, F)
        y      : np.ndarray (N,)
        mean   : np.ndarray (F,)  — pass training mean for val/test
        std    : np.ndarray (F,)  — pass training std  for val/test
        augment: bool             — enable data augmentation
        """
        # Compute or apply normalization
        if mean is None:
            self.mean = X.mean(axis=(0, 1))          # (F,)
            self.std  = X.std(axis=(0, 1)) + 1e-8    # (F,)
        else:
            self.mean = mean
            self.std  = std

        X_norm = (X - self.mean) / self.std          # (N, T, F)

        self.X       = torch.tensor(X_norm, dtype=torch.float32)
        self.y       = torch.tensor(y,      dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()   # (T, F)

        if self.augment:
            # Gaussian noise
            x = x + torch.randn_like(x) * NOISE_STD

            # Random frame dropout
            mask = torch.rand(x.shape[0]) < FRAME_DROP
            x[mask] = 0.0

        return x, self.y[idx]


def get_dataloaders(npz_file=NPZ_FILE, batch_size=BATCH_SIZE):
    """
    Loads keypoints.npz and returns three DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names
    """
    data = np.load(npz_file, allow_pickle=True)
    X       = data["X"]        # (N, T, F)
    y       = data["y"]        # (N,)
    splits  = data["splits"]   # (N,)  'train'/'val'/'test'
    glosses = data["glosses"]  # (100,)

    X_train = X[splits == "train"]
    y_train = y[splits == "train"]
    X_val   = X[splits == "val"]
    y_val   = y[splits == "val"]
    X_test  = X[splits == "test"]
    y_test  = y[splits == "test"]

    print(f"Loaded {npz_file}")
    print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")

    train_ds = ASLDataset(X_train, y_train, augment=True)
    val_ds   = ASLDataset(X_val,   y_val,   mean=train_ds.mean, std=train_ds.std)
    test_ds  = ASLDataset(X_test,  y_test,  mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, list(glosses)


# ── MLP Baseline ──────────────────────────────────────────────────────────────
class MLPBaseline(nn.Module):
    """
    Simple 3-layer MLP baseline.

    Input:  (B, T, F)  — keypoint sequence
    Step 1: Temporal mean pooling  →  (B, F)
    Step 2: MLP [F → 256 → 128 → 64 → N_CLASSES]
    Output: (B, N_CLASSES) logits

    No temporal modeling — all frame order information is discarded.
    This is intentional: the baseline exists to show that keypoints
    carry signal above chance even without sequence modeling.
    """

    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.mean(dim=1)   # temporal mean pooling → (B, F)
        return self.mlp(x)  # (B, N_CLASSES)


# ── Evaluation helper ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct_top1 = correct_top5 = total = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                         # (B, C)

        # Top-1
        pred_top1 = logits.argmax(dim=-1)
        correct_top1 += (pred_top1 == y).sum().item()

        # Top-5
        top5 = logits.topk(5, dim=-1).indices    # (B, 5)
        correct_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

        total += y.size(0)
        all_preds.extend(pred_top1.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    top1 = correct_top1 / total * 100
    top5 = correct_top5 / total * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    return top1, top5, macro_f1


# ── Training loop ─────────────────────────────────────────────────────────────
def train_baseline(train_loader, val_loader, device=DEVICE, epochs=EPOCHS, lr=LR):
    model = MLPBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining MLP Baseline on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Top-1':>9}  {'Val Top-5':>9}  {'Val F1':>7}")
    print("-" * 55)

    best_val_top1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_top1, val_top5, val_f1 = evaluate(model, val_loader, device)

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), "mlp_baseline_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg_loss:>10.4f}  {val_top1:>8.2f}%  {val_top5:>8.2f}%  {val_f1:>6.2f}%")

    print(f"\nBest validation Top-1: {best_val_top1:.2f}%")
    print(f"Random chance baseline: {100/N_CLASSES:.2f}%")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    model = train_baseline(train_loader, val_loader)

    # Final test set evaluation
    model.load_state_dict(torch.load("mlp_baseline_best.pt", map_location=DEVICE))
    test_top1, test_top5, test_f1 = evaluate(model, test_loader, DEVICE)

    print("\n── Test Set Results (MLP Baseline) ──")
    print(f"  Top-1 Accuracy : {test_top1:.2f}%")
    print(f"  Top-5 Accuracy : {test_top5:.2f}%")
    print(f"  Macro F1-Score : {test_f1:.2f}%")
    print(f"  Random chance  : {100/N_CLASSES:.2f}%")
