import warnings
import inspect
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import math
import re

warnings.filterwarnings("ignore", category=FutureWarning, message=r"You are using `torch\.load` with `weights_only=False`.*")

print("asdf")

LAB_NAMES: List[str] = [
    # CBC labs
    "creatinine_mgdl", "hemoglobin_gdl", "sodium_mEqL",
    "wbc_kul", "platelet_kul", "bun_mgdl", 
    # Add CBC and BMP panels
    "rbc_million_ul", "hematocrit_percent", "mcv_fl", "mch_pg", "mchc_gdl", "rdw_percent",
    "potassium_mEqL", "chloride_mEqL", "bicarbonate_mEqL", "glucose_mgdl", "calcium_mgdl"
]

VOCAB: List[str] = [
    "normal", "mild", "moderate", "severe",
    "renal_failure", "anemia", "infection", "dehydration",
    "hypernatremia", "hyponatremia", "thrombocytopenia", "liver_failure", "hypoglycemia", "hypertension"
]

VOCAB_SIZE = len(VOCAB)
TOKEN2IDX = {tok: i for i, tok in enumerate(VOCAB)}

SEVERITY_MAP = {
    "normal": 1.0,
    "mild": 0.5,
    "moderate": 1.0,
    "severe": 1.5
}

class FakeMultiLabDataset(Dataset):
    def __init__(self, n_samples: int = 32000, img_size: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.imgs = rng.normal(0.0, 1.0, size=(n_samples, 1, img_size, img_size)).astype(np.float32)
        self.labs, self.bows, self.notes = self._gen_targets(self.imgs, rng)

    def _extract_severity_and_condition(self, text: str) -> Tuple[str, str]:
        severity_match = re.search(r"\b(mild|moderate|severe|normal)\b", text.lower())
        severity = severity_match.group(0) if severity_match else "normal"
        condition_match = re.search(r"\b(renal_failure|anemia|infection|dehydration|thrombocytopenia|hypernatremia|hyponatremia|liver_failure|hypoglycemia|hypertension)\b", text.lower())
        condition = condition_match.group(0) if condition_match else "normal"
        return severity, condition

    def _gen_targets(self, imgs: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        n = imgs.shape[0]
        labs = np.zeros((n, len(LAB_NAMES)), dtype=np.float32)
        bows = np.zeros((n, VOCAB_SIZE), dtype=np.float32)
        notes = []

        for i in range(n):
            sev = rng.choice(["mild", "moderate", "severe", "normal"], p=[0.2, 0.3, 0.3, 0.2])
            cond = rng.choice(["renal_failure", "anemia", "infection", "dehydration", "thrombocytopenia", "hypernatremia", "hyponatremia", "liver_failure", "hypoglycemia", "hypertension"], size=1)[0]
            note = f"{sev} {cond}"
            severity, condition = self._extract_severity_and_condition(note)

            base = np.array([1.0, 14.0, 140.0, 7.0, 250.0, 12.0, 4.5, 40.0, 90.0, 30.0, 33.0, 13.0, 5.0, 100.0, 24.0, 110.0, 9.0])  # Updated for full panel
            delta = np.zeros_like(base)

            if condition == "renal_failure":
                delta[0] += np.random.normal(2.0, 0.1)
                delta[5] += np.random.normal(8.0, 0.3)
            if condition == "anemia":
                delta[1] -= np.random.normal(4.0, 0.2)
            if condition == "infection":
                delta[3] += np.random.normal(4.0, 0.3)
            if condition == "dehydration":
                delta[0] += np.random.normal(0.5, 0.05)
                delta[2] += np.random.normal(5.0, 0.2)
                delta[5] += np.random.normal(2.0, 0.2)
            if condition == "hypernatremia":
                delta[2] += np.random.normal(8.0, 0.3)
            if condition == "hyponatremia":
                delta[2] -= np.random.normal(8.0, 0.3)
            if condition == "thrombocytopenia":
                delta[4] -= np.random.normal(150.0, 5.0)
            if condition == "liver_failure":
                delta[1] -= np.random.normal(3.0, 0.1)
                delta[9] += np.random.normal(20.0, 0.3)
            if condition == "hypoglycemia":
                delta[13] -= np.random.normal(15.0, 0.4)
            if condition == "hypertension":
                delta[2] += np.random.normal(5.0, 0.3)
                delta[4] += np.random.normal(30.0, 0.5)

            labs[i] = base + delta * SEVERITY_MAP[severity] + rng.normal(0, 0.2, size=len(base))

            notes.append(note)
            for tok in [severity, condition]:
                bows[i, TOKEN2IDX[tok]] = 1.0

        return labs, bows, notes

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.imgs[idx]),
            torch.from_numpy(self.bows[idx]),
            torch.from_numpy(self.labs[idx]),
            self.notes[idx]
        )

class SmallCNNText(nn.Module):
    def __init__(self, n_labs: int, vocab_size: int):
        super().__init__()
        self.img_enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.txt_enc = nn.Sequential(nn.Linear(vocab_size, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, n_labs))

    def forward(self, img, bow):
        return self.head(torch.cat([self.img_enc(img).flatten(1), self.txt_enc(bow)], dim=1))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    abs_err = torch.zeros(len(LAB_NAMES), device=device)
    n = 0
    for img, bow, lab, _ in loader:
        img, bow, lab = img.to(device), bow.to(device), lab.to(device)
        abs_err += (model(img, bow) - lab).abs().sum(0)
        n += lab.size(0)
    return (abs_err / n).cpu()

def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    running = 0.0
    for img, bow, lab, _ in loader:
        img, bow, lab = img.to(device), bow.to(device), lab.to(device)
        optim.zero_grad()
        loss = loss_fn(model(img, bow), lab)
        loss.backward()
        optim.step()
        running += loss.item() * img.size(0)
    return running / len(loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = FakeMultiLabDataset()
    train_set, val_set = torch.utils.data.random_split(full, [25600, 6400])  # Updated for more data
    train_loader = DataLoader(train_set, 64, shuffle=True)
    val_loader = DataLoader(val_set, 64)

    model = SmallCNNText(len(LAB_NAMES), VOCAB_SIZE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_losses = []
    val_maes = []

    def custom_loss_fn(pred, target):
        return (pred - target).abs() ** 2.2

    loss_fn = lambda pred, target: custom_loss_fn(pred, target).mean()

    for ep in range(1, 101):
        train_loss = train_one_epoch(model, train_loader, optim, loss_fn, device)
        val_mae = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_maes.append(val_mae.mean().item())

        print(f"\nEpoch {ep:02d} Results")
        print("------------------------------")
        print(f"Train L1 Loss: {train_loss:.4f} (log10={math.log10(train_loss + 1e-8):.4f})")
        print(f"Val Mean Absolute Error: {val_mae.mean():.4f} (log10={math.log10(val_mae.mean().item() + 1e-8):.4f})")
        for name, val in zip(LAB_NAMES, val_mae):
            print(f"  {name:>16}: MAE = {val:.4f}")

    plt.plot([math.log10(l + 1e-8) for l in train_losses], label="Train Loss (log10)")
    plt.plot([math.log10(m + 1e-8) for m in val_maes], label="Val MAE (log10)")
    plt.xlabel("Epoch")
    plt.ylabel("Log-scaled Loss")
    plt.legend()
    plt.title("Log Training Curve")
    plt.grid(True)
    plt.savefig("log_loss_curve.png")
    print("\nSaved training curve to log_loss_curve.png\n")

    ckpt = "multimodal_small_cnn.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\nWeights saved → {ckpt}\n")

    reloaded = SmallCNNText(len(LAB_NAMES), VOCAB_SIZE).to(device)
    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    reloaded.load_state_dict(torch.load(ckpt, **load_kwargs))
    reloaded.eval()

    print("--- Demo predictions (reloaded model) ---")
    with open("demo_lab_predictions.txt", "w", encoding='utf-8') as f:
        for i in range(10):
            img, bow, true_lab, note = val_set[i]
            pred = reloaded(img.unsqueeze(0).to(device), bow.unsqueeze(0).to(device)).squeeze(0).cpu()
            f.write(f"Note: {note}\n")
            print(f"Note: {note}")
            for name, t, p in zip(LAB_NAMES, true_lab, pred):
                pct_diff = 100.0 * abs(p.item() - t.item()) / (abs(t.item()) + 1e-6)
                line = f"  {name:>15}: true={t:.2f} | pred={p:.2f} | Δ%={pct_diff:.1f}%"
                print(line)
                f.write(line + "\n")
            print("-")
            f.write("-\n")

if __name__ == "__main__":
    main()
