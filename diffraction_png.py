"""
src/diffraction_png.py

Render diffraction patterns to PNG.

Single crystal (all 8 angles):
  python src/diffraction_png.py data/cif_cache/NaCl.cif
  python src/diffraction_png.py data/cif_cache/NaCl.cif out.png
  python src/diffraction_png.py data/cif_cache/NaCl.cif --single

Class overview — ONE random example per class x all 8 angles (9 rows x 8 cols):
  python src/diffraction_png.py --overview
  python src/diffraction_png.py --overview --seed 7

"""
import os, sys, glob, argparse
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gemmi

from src.multi_angle_simulator import (
    _precompute_hkl_and_intensities, _project_crystal, crystal_rotation)
from src.regenerate_xrd2d import (ORIENTATIONS, system_id, SYSTEMS,
                                  make_amorphous_views, make_noise_views, add_detector_noise)


IMG, DMIN, SCALE, SIGMA, MININT = 256, 1.2, 130.0, 1.6, 0.005
DS_IMG, DS_DMIN, DS_SCALE, DS_SIGMA, DS_MININT = 128, 2.0, 110.0, 1.4, 0.01


def render_views(cif_path, orientations=ORIENTATIONS,
                 img=IMG, dmin=DMIN, scale=SCALE, sigma=SIGMA, minint=MININT, noise_rng=None):
    sm = gemmi.read_small_structure(str(cif_path))
    hkl, I, B = _precompute_hkl_and_intensities(sm, dmin=dmin)
    views = []
    for (phi, chi, om) in orientations:
        im = _project_crystal(hkl, I, B, crystal_rotation(phi, chi, om), img, scale, sigma, minint)
        mx = im.max()
        if mx > 0:
            im = im / mx
        if noise_rng is not None:
            im = add_detector_noise(im, noise_rng)
        views.append(im)
    return views, sm


def save_diffraction_png(cif_path, out_path=None, single=False):
    views, sm = render_views(cif_path, orientations=([ORIENTATIONS[1]] if single else ORIENTATIONS))
    name = os.path.splitext(os.path.basename(str(cif_path)))[0]
    sgn = sm.spacegroup.number if sm.spacegroup else None
    sys_name = SYSTEMS[system_id(sgn)] if sgn else "unknown"
    if out_path is None:
        out_path = os.path.join(PROJECT_ROOT, "outputs", "diffraction", f"{name}_diffraction.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if single:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.asarray(views[0]), cmap="inferno"); ax.axis("off")
    else:
        n = len(views); cols = 4; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
        for i, ax in enumerate(np.atleast_1d(axes).flat):
            if i < n:
                ax.imshow(np.asarray(views[i]), cmap="inferno")
                ax.set_title(f"orientation {i+1}", fontsize=8, color="0.8")
            ax.axis("off")
    sg_txt = f"space group #{sgn}" if sgn else "space group unknown"
    fig.suptitle(f"{name}   |   {sys_name}   |   {sg_txt}", color="white", fontsize=11)
    fig.patch.set_facecolor("black"); plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="black", bbox_inches="tight"); plt.close(fig)
    return out_path


def _pick_one_cif_per_system(cache_dir, rng):
    """Return {system_id: cif_path} — one random crystal per system (0..6)."""
    cifs = glob.glob(os.path.join(cache_dir, "*.cif"))
    rng.shuffle(cifs)
    picked = {}
    for f in cifs:
        if len(picked) == 7:
            break
        try:
            sm = gemmi.read_small_structure(f); c = sm.cell
            if sm.spacegroup is None or max(c.a, c.b, c.c) > 20: continue
            s = system_id(sm.spacegroup.number)
            if s not in picked:
                picked[s] = f
        except Exception:
            continue
    return picked


def save_class_overview(out_path=None, seed=0, cache_dir=None):
    """
    ONE figure: a random example of every class (rows) across all 8 angles (cols).
    Rows 0-6 = crystal systems (random CIF each, dataset-faithful render incl.
    detector noise); row 7 = amorphous; row 8 = noise.
    """
    rng = np.random.default_rng(seed)
    nrng = np.random.default_rng(seed + 100)
    cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "data", "cif_cache")
    if out_path is None:
        out_path = os.path.join(PROJECT_ROOT, "outputs", "diffraction", "class_overview.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    picked = _pick_one_cif_per_system(cache_dir, rng)
    n_ang = len(ORIENTATIONS)
    rows = []   # (label, [8 view arrays])
    for s in range(7):
        f = picked.get(s)
        if f is None:
            rows.append((SYSTEMS[s] + " (none)", [np.zeros((DS_IMG, DS_IMG))] * n_ang)); continue
        views, _ = render_views(f, img=DS_IMG, dmin=DS_DMIN, scale=DS_SCALE,
                                sigma=DS_SIGMA, minint=DS_MININT, noise_rng=nrng)
        rows.append((f"{SYSTEMS[s]}\n{os.path.splitext(os.path.basename(f))[0]}", views))
    rows.append((SYSTEMS[7], list(make_amorphous_views(rng, n_ang, DS_IMG, DS_SCALE).astype(np.float32) / 255.0)))
    rows.append((SYSTEMS[8], list(make_noise_views(rng, n_ang, DS_IMG).astype(np.float32) / 255.0)))

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, n_ang, figsize=(n_ang * 1.4, nrows * 1.5))
    for ri, (label, views) in enumerate(rows):
        for ci in range(n_ang):
            ax = axes[ri, ci]
            ax.imshow(np.asarray(views[ci]), cmap="inferno"); ax.axis("off")
            if ri == 0:
                ax.set_title(f"angle {ci+1}", fontsize=8, color="0.8")
        axes[ri, 0].set_ylabel(label, fontsize=8, color="white", rotation=0,
                               ha="right", va="center", labelpad=38)
        axes[ri, 0].axis("on"); axes[ri, 0].set_xticks([]); axes[ri, 0].set_yticks([])
        for sp in axes[ri, 0].spines.values(): sp.set_visible(False)
    fig.suptitle("Diffraction patterns — one random example per class x all 8 angles",
                 color="white", fontsize=12)
    fig.patch.set_facecolor("black"); plt.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="black", bbox_inches="tight"); plt.close(fig)
    return out_path

def _save_row(views, title, out_path):
    """Save one class's 8 angle-views as a single horizontal-strip PNG."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(views)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.6, 2.1))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        ax.imshow(np.asarray(views[i]), cmap="inferno")
        ax.set_title(f"angle {i+1}", fontsize=7, color="0.8"); ax.axis("off")
    fig.suptitle(title, color="white", fontsize=11)
    fig.patch.set_facecolor("black"); plt.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="black", bbox_inches="tight"); plt.close(fig)
    return out_path


def save_per_class_pngs(out_dir=None, seed=0, cache_dir=None):
    """
    Save ONE PNG per class (9 files), each showing a random example of that
    class across all 8 angles. Crystal rows use a random real CIF of that
    system (dataset-faithful render incl. detector noise).
    """
    rng = np.random.default_rng(seed); nrng = np.random.default_rng(seed + 100)
    cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "data", "cif_cache")
    out_dir = out_dir or os.path.join(PROJECT_ROOT, "outputs", "diffraction", "per_class")
    os.makedirs(out_dir, exist_ok=True)
    picked = _pick_one_cif_per_system(cache_dir, rng)
    paths = []
    for s in range(7):
        f = picked.get(s)
        if f is None:
            print(f"  (no CIF found for {SYSTEMS[s]})"); continue
        views, _ = render_views(f, img=DS_IMG, dmin=DS_DMIN, scale=DS_SCALE,
                                sigma=DS_SIGMA, minint=DS_MININT, noise_rng=nrng)
        name = os.path.splitext(os.path.basename(f))[0]
        paths.append(_save_row(views, f"{SYSTEMS[s]}  —  {name}",
                               os.path.join(out_dir, f"{s}_{SYSTEMS[s]}.png")))
    av = list(make_amorphous_views(rng, len(ORIENTATIONS), DS_IMG, DS_SCALE).astype(np.float32) / 255.0)
    paths.append(_save_row(av, SYSTEMS[7], os.path.join(out_dir, "7_amorphous.png")))
    nv = list(make_noise_views(rng, len(ORIENTATIONS), DS_IMG).astype(np.float32) / 255.0)
    paths.append(_save_row(nv, SYSTEMS[8], os.path.join(out_dir, "8_noise.png")))
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cif", nargs="?", default=None, help="path to a CIF (omit with --overview)")
    ap.add_argument("out", nargs="?", default=None, help="output PNG path (optional)")
    ap.add_argument("--single", action="store_true", help="render one pattern instead of 8")
    ap.add_argument("--overview", action="store_true", help="one random example per class x 8 angles")
    ap.add_argument("--per-class", action="store_true", help="save each class as its own PNG (9 files)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.per_class:
        for pth in save_per_class_pngs(out_dir=a.out, seed=a.seed):
            print("  wrote", pth)
    elif a.overview:
        print("  wrote", save_class_overview(out_path=a.out, seed=a.seed))
    elif a.cif:
        print("  wrote", save_diffraction_png(a.cif, a.out, a.single))
    else:
        ap.error("give a CIF path, or use --overview")
