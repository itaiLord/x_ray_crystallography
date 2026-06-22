"""
src/train_classifier.py
Diffraction-pattern classifier (9 classes: 7 crystal systems + amorphous + noise).


Run:
  python src/train_classifier.py                 # resume if a checkpoint exists
  python src/train_classifier.py --fresh         # ignore checkpoint, start over
  python src/train_classifier.py --epochs 40     # run 40 MORE epochs
  python src/train_classifier.py --test          # evaluate on held-out TEST split

Splitting (3-way, by a STABLE HASH of each crystal_id)
------------------------------------------------------
  train (70%) | val (15%) checkpoint selection | test (15%) --test only.
"""
import os, sys, argparse, glob, hashlib
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.regenerate_xrd2d import SYSTEMS, ORIENTATIONS, system_id
from src.multi_angle_simulator import (
    _precompute_hkl_and_intensities, _project_crystal, crystal_rotation)

N_CLASS = len(SYSTEMS)
N_CRYSTAL_SYS = 7   # classes 0..6 are the crystal systems; 7=amorphous, 8=noise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_BEST = os.path.join(PROJECT_ROOT, "checkpoints", "classifier_best.pt")
CKPT_LAST = os.path.join(PROJECT_ROOT, "checkpoints", "classifier_last.pt")
IMG, DMIN, SCALE, SIGMA, MININT = 128, 2.0, 110.0, 1.4, 0.01


class ViewDataset(Dataset):
    def __init__(self, images, labels):
        self.x = images; self.y = labels
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        img = self.x[i].astype(np.float32) / 255.0
        return torch.from_numpy(img)[None], int(self.y[i])


def crystal_hash_split(crystal_ids, val_frac=0.15, test_frac=0.15, seed=0):
    """
    Split crystals into train/val/test by a STABLE HASH of each crystal_id.
    A given crystal ALWAYS lands in the same split, regardless of dataset size
    or order. So you can add more CIFs and resume training without ever leaking
    a previously-trained crystal into the test set (the bug this replaced:
    index-based splitting reshuffled assignments whenever the data changed).
    """
    salt = f"xrd2d_v1_{seed}"
    tr, va, te = [], [], []
    for i, cid in enumerate(crystal_ids):
        h = int(hashlib.md5(f"{salt}|{cid}".encode()).hexdigest(), 16) % 100000 / 100000.0
        if   h < test_frac:            te.append(i)
        elif h < test_frac + val_frac: va.append(i)
        else:                          tr.append(i)
    return np.array(tr), np.array(va), np.array(te)


def expand_to_views(images_4d, labels, crystal_idx):
    sub = images_4d[crystal_idx]
    K, V = sub.shape[0], sub.shape[1]
    x = sub.reshape(K * V, sub.shape[2], sub.shape[3])
    y = np.repeat(labels[crystal_idx], V)
    return x, y, V


class DiffractionCNN(nn.Module):
    def __init__(self, n_class=N_CLASS):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_class))
    def forward(self, x):
        return self.head(self.features(x))


def confusion(y_true, y_pred, n=N_CLASS):
    m = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m

def per_class_f1(cm):
    f1 = np.zeros(N_CLASS)
    for c in range(N_CLASS):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1[c] = 2 * p * r / (p + r) if p + r else 0.0
    return f1

def macro_f1(cm):
    return float(per_class_f1(cm).mean())

def crystal_only_f1(cm):
    """macro-F1 over the 7 crystal-system classes that have support (excl. amorphous/noise)."""
    f1 = per_class_f1(cm); support = cm.sum(1)
    present = [c for c in range(N_CRYSTAL_SYS) if support[c] > 0]
    return float(np.mean([f1[c] for c in present])) if present else float("nan")

def print_report(cm, title="report"):
    f1 = per_class_f1(cm); support = cm.sum(1)
    acc = np.trace(cm) / max(cm.sum(), 1)
    print(f"  {title}")
    print("    class           P      R      F1    support")
    for c in range(N_CLASS):
        if support[c] == 0 and cm[:, c].sum() == 0:
            continue
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"    {SYSTEMS[c]:13s} {p:5.2f}  {r:5.2f}  {f1[c]:5.2f}   {int(support[c]):5d}")
    print(f"    accuracy={acc:.3f}   macro-F1(9)={f1.mean():.3f}   "
          f"crystal-systems macro-F1(7)={crystal_only_f1(cm):.3f}")
    return acc, f1.mean()

def print_confusion(cm):
    print("  confusion (rows=true, cols=pred):")
    print("      " + " ".join(f"{s[:4]:>5s}" for s in SYSTEMS))
    for c in range(N_CLASS):
        print(f"  {SYSTEMS[c][:4]:>4s} " + " ".join(f"{cm[c, j]:5d}" for j in range(N_CLASS)))


def render_cif_views(cif_path):
    import gemmi
    sm = gemmi.read_small_structure(str(cif_path))
    sg = sm.spacegroup
    hkl, I, B = _precompute_hkl_and_intensities(sm, dmin=DMIN)
    views = []
    for (phi, chi, om) in ORIENTATIONS:
        R = crystal_rotation(phi, chi, om)
        im = _project_crystal(hkl, I, B, R, IMG, SCALE, SIGMA, MININT)
        mx = im.max()
        if mx > 0:
            im = im / mx
        views.append(im.astype(np.float32))
    return np.stack(views), (sg.number if sg else None)



def load_model(ckpt_path):
    model = DiffractionCNN().to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def run_test(args):
    if not os.path.exists(CKPT_BEST):
        print(f"  No checkpoint at {CKPT_BEST} — train first."); return
    model, ck = load_model(CKPT_BEST)
    print(f"  Loaded {CKPT_BEST}  (epoch {ck.get('epoch','?')}, "
          f"val macro-F1 {ck.get('macro_f1', float('nan')):.3f})")


    if args.cif:
        views, sg = render_cif_views(args.cif)
        x = torch.from_numpy(views)[:, None].to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).mean(0).cpu().numpy()
        pred = int(probs.argmax())
        print(f"\n  CIF: {os.path.basename(args.cif)}")
        print(f"  Predicted : {SYSTEMS[pred]}  (confidence {probs[pred]:.2f})")
        if sg is not None:
            true = system_id(sg)
            print(f"  True      : {SYSTEMS[true]}  (space group #{sg})  -> "
                  f"{'CORRECT' if true == pred else 'WRONG'}")
        order = np.argsort(probs)[::-1]
        print("  Top-3: " + ", ".join(f"{SYSTEMS[i]} {probs[i]:.2f}" for i in order[:3]))
        return

    with h5py.File(args.h5, "r") as h:
        images = np.array(h["images"]); labels = np.array(h["labels"])
        crystal_ids = [x.decode() if hasattr(x, "decode") else str(x) for x in h["crystal_ids"][:]]
    _, _, te_c = crystal_hash_split(crystal_ids, seed=args.seed)
    xte, yte, V = expand_to_views(images, labels, te_c)
    loader = DataLoader(ViewDataset(xte, yte), batch_size=256, shuffle=False)
    probs_all = []
    with torch.no_grad():
        for x, _ in loader:
            probs_all.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    probs_all = np.concatenate(probs_all)
    yp = probs_all.argmax(1)

    print(f"\n  === HELD-OUT TEST split: {len(te_c)} crystals, {len(yte)} views ===")
    print(f"  (selection never saw these)")
    print_report(confusion(yte, yp), "per-VIEW")

    K = len(te_c)
    crys_pred = probs_all.reshape(K, V, N_CLASS).mean(1).argmax(1)
    cm_crys = confusion(labels[te_c], crys_pred)
    print()
    print_report(cm_crys, "per-CRYSTAL (majority over 8 views — real-world metric)")
    print()
    print_confusion(cm_crys)



def main(args):
    with h5py.File(args.h5, "r") as h:
        images = np.array(h["images"]); labels = np.array(h["labels"])
        crystal_ids = [x.decode() if hasattr(x, "decode") else str(x) for x in h["crystal_ids"][:]]
    print(f"  Device : {device}")
    counts = np.bincount(labels, minlength=N_CLASS)
    print("  Per-class crystals: " + ", ".join(f"{SYSTEMS[c]}={counts[c]}" for c in range(N_CLASS)))

    tr_c, va_c, te_c = crystal_hash_split(crystal_ids, seed=args.seed)
    xtr, ytr, _ = expand_to_views(images, labels, tr_c)
    xva, yva, _ = expand_to_views(images, labels, va_c)
    print(f"  Crystals  train/val/test: {len(tr_c)}/{len(va_c)}/{len(te_c)}  "
          f"(test held out for --test)")

    tr_loader = DataLoader(ViewDataset(xtr, ytr), batch_size=args.bs, shuffle=True)
    va_loader = DataLoader(ViewDataset(xva, yva), batch_size=args.bs, shuffle=False)

    tr_counts = np.bincount(labels[tr_c], minlength=N_CLASS).astype(np.float64)
    w = tr_counts.sum() / (N_CLASS * np.maximum(tr_counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    model = DiffractionCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=6, min_lr=1e-6)

    start_epoch, best_f1 = 0, -1.0
    os.makedirs(os.path.dirname(CKPT_LAST), exist_ok=True)
    if not args.fresh and os.path.exists(CKPT_LAST):
        ck = torch.load(CKPT_LAST, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            try: sched.load_state_dict(ck["scheduler_state_dict"])
            except Exception: print("  (scheduler type changed — starting its schedule fresh)")
        start_epoch = ck.get("epoch", 0); best_f1 = ck.get("best_f1", -1.0)
        print(f"  Resumed from {CKPT_LAST} (epoch {start_epoch}, best val macro-F1 {best_f1:.3f})")
    elif not args.fresh and os.path.exists(CKPT_BEST):
        ck = torch.load(CKPT_BEST, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        best_f1 = ck.get("macro_f1", -1.0); start_epoch = ck.get("epoch", 0)
        print(f"  Continuing from best ckpt (val macro-F1 {best_f1:.3f}); fresh optimizer.")
    else:
        print("  Starting fresh." if args.fresh else "  No checkpoint — starting fresh.")

    for ep in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train(); tl = 0.0; nb = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1

        model.eval(); yt, yp = [], []
        with torch.no_grad():
            for x, y in va_loader:
                yp.append(model(x.to(device)).argmax(1).cpu().numpy()); yt.append(y.numpy())
        cm = confusion(np.concatenate(yt), np.concatenate(yp))
        acc = np.trace(cm) / max(cm.sum(), 1); f1 = macro_f1(cm)
        sched.step(f1)

        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "epoch": ep, "best_f1": float(best_f1)}, CKPT_LAST)
        flag = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": ep, "macro_f1": float(f1)}, CKPT_BEST)
            flag = "  <- best"
        print(f"  Ep {ep:>3}  tr-loss {tl/max(nb,1):.4f}  val-acc {acc:.3f}  "
              f"val macro-F1 {f1:.3f}  lr {opt.param_groups[0]['lr']:.2e}{flag}")

    print(f"\n  Best val macro-F1: {best_f1:.3f}   ckpt: {CKPT_BEST}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=os.path.join(PROJECT_ROOT, "datasets", "xrd_2d.h5"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--cif", default=None)
    ap.add_argument("--real", default=None, help="dir of real images: <dir>/<class_name>/*.png")
    args = ap.parse_args()
    if args.test:
        run_test(args)
    else:
        main(args)
