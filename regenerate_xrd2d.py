"""
src/regenerate_xrd2d.py
=======================
Build datasets/xrd_2d.h5 for the classifier, OFFLINE from data/cif_cache/.

Classes (9): 7 crystal systems + amorphous (glass/liquid) + noise/air.
  Crystals  -> 2D diffraction images via multi_angle_simulator (sharp spots).
  Amorphous -> simulated diffuse halos (glass / liquid / plastic): no Bragg
               peaks, just a broad ring at the nearest-neighbour distance.

Run:
  python src/regenerate_xrd2d.py                  # crystals + 450 amorphous
  python src/regenerate_xrd2d.py --amorphous 0    # crystals only
  python src/regenerate_xrd2d.py --amorphous 700

Output h5:
  images       (N, n_views, 64, 64) uint8
  labels       (N,)  int 0..7   (triclinic..cubic, 7=amorphous)
  crystal_ids  (N,)  str
"""
import os, sys, glob, argparse
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import numpy as np, h5py, gemmi
from src.multi_angle_simulator import (
    _precompute_hkl_and_intensities, _project_crystal, crystal_rotation)

SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
           "trigonal", "hexagonal", "cubic", "amorphous", "noise"]
AMORPHOUS_LABEL = 7
NOISE_LABEL = 8
ORIENTATIONS = [(0,0,0),(35,15,25),(70,-20,55),(110,10,95),
                (150,-15,130),(25,25,160),(95,-10,200),(200,5,250)]

def system_id(num):
    if num <= 2:   return 0
    if num <= 15:  return 1
    if num <= 74:  return 2
    if num <= 142: return 3
    if num <= 167: return 4
    if num <= 194: return 5
    return 6


def make_amorphous_views(rng, n_views=8, img=64, scale=44.0):
    """
    Simulate an amorphous / liquid / glass diffraction pattern: broad diffuse
    halo(s) instead of sharp Bragg spots. The first (and strongest) halo sits
    at q0 = 1/d_nn, where d_nn is the nearest-neighbour distance (~1.8-3.3 A).
    Returns (n_views, img, img) uint8 — n_views noisy realisations of one
    'material' (amorphous scattering is isotropic, so views differ only by noise).
    """
    cy = cx = img / 2.0
    yy, xx = np.mgrid[0:img, 0:img]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    d_nn  = rng.uniform(1.8, 3.3)
    r0    = (1.0 / d_nn) * scale
    width = rng.uniform(3.0, 7.0)
    second = rng.random() < 0.5

    views = []
    for _ in range(n_views):
        r0j = r0 * rng.uniform(0.95, 1.05)
        ring = np.exp(-((r - r0j) / width) ** 2)
        if second:
            ring = ring + 0.4 * np.exp(-((r - 1.7 * r0j) / (width * 1.3)) ** 2)
        ring[r < 3] = 0.0                                  # beamstop
        ring = np.clip(ring + rng.normal(0, 0.03, ring.shape), 0, None)
        mx = ring.max()
        if mx > 0:
            ring = ring / mx
        views.append((ring * 255).astype(np.uint8))
    return np.stack(views)

def make_noise_views(rng, n_views=8, img=64):
    """
    Simulate an empty-beam / air-scatter / pure-noise frame: NO Bragg spots and
    NO diffraction ring -- just a faint central glow (direct-beam + air scatter)
    plus broadband detector speckle and a few hot pixels. This is the 'no usable
    sample' reject class. Returns (n_views, img, img) uint8.
    """
    cy = cx = img / 2.0
    yy, xx = np.mgrid[0:img, 0:img]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    views = []
    for _ in range(n_views):
        glow  = np.exp(-(r / rng.uniform(8, 20)) ** 2) * rng.uniform(0.1, 0.4)
        speck = rng.random((img, img)) ** rng.uniform(2.0, 5.0)   # sparse speckle
        frame = glow + 0.6 * speck
        for _ in range(int(rng.integers(0, 6))):                  # hot pixels
            frame[int(rng.integers(0, img)), int(rng.integers(0, img))] += rng.uniform(0.5, 1.0)
        mx = frame.max()
        if mx > 0:
            frame = frame / mx
        views.append((frame * 255).astype(np.uint8))
    return np.stack(views)

def add_detector_noise(img01, rng, bg=0.004, read=0.008, poisson_scale=600.0):
    """
    Add realistic detector/air noise to a normalised [0,1] pattern WITHOUT
    burying faint Bragg spots.

    Key: shot (Poisson) noise scales WITH the local signal — that's the real
    physics — so noise lands mostly where there is signal, while the dark gaps
    between spots stay dark. Only a tiny flat air-scatter floor (bg) and a small
    Gaussian read noise are added everywhere.

    The previous version added a large FLAT background (0.02) across the whole
    frame, which swamped low-intensity reflections (~4.6x faint/background
    contrast). This version keeps ~20x contrast, so weak spots survive.
    Returns a re-normalised [0,1] float array. Applied to EVERY crystal image.
    """
    x = rng.poisson(np.clip(img01, 0, None) * poisson_scale) / poisson_scale
    x = x + bg + rng.normal(0, read, x.shape)
    x = np.clip(x, 0, None)
    mx = x.max()
    return (x / mx) if mx > 0 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(PROJECT_ROOT, "data", "cif_cache"))
    ap.add_argument("--out",   default=os.path.join(PROJECT_ROOT, "datasets", "xrd_2d.h5"))
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--dmin", type=float, default=2.0)
    ap.add_argument("--scale", type=float, default=110.0)
    ap.add_argument("--sigma", type=float, default=1.4)
    ap.add_argument("--min-int", type=float, default=0.01)
    ap.add_argument("--max-edge", type=float, default=20.0)
    ap.add_argument("--cap", type=int, default=470, help="max crystals per system (balances classes)")
    ap.add_argument("--no-detector-noise", action="store_true", help="disable per-image detector noise")
    ap.add_argument("--amorphous", type=int, default=450, help="# synthetic amorphous samples")
    ap.add_argument("--noise", type=int, default=450, help="# synthetic noise/air samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cifs = sorted(glob.glob(os.path.join(a.cache, "*.cif")))
    print(f"  CIFs in cache: {len(cifs)}")
    IMGS, LAB, IDS = [], [], []
    skipped = 0
    noise_rng = np.random.default_rng(a.seed + 2)
    per_sys = np.zeros(7, dtype=int)
    for f in cifs:
        cid = os.path.splitext(os.path.basename(f))[0]
        try:
            sm = gemmi.read_small_structure(f); c = sm.cell; sg = sm.spacegroup
            if sg is None: skipped += 1; continue
            if min(c.a,c.b,c.c) <= 0 or max(c.a,c.b,c.c) > a.max_edge: skipped += 1; continue
            hkl, I, B = _precompute_hkl_and_intensities(sm, dmin=a.dmin)
            if len(hkl) == 0: skipped += 1; continue
            views = []
            for (phi, chi, om) in ORIENTATIONS:
                R = crystal_rotation(phi, chi, om)
                im = _project_crystal(hkl, I, B, R, a.img, a.scale, a.sigma, a.min_int)
                mx = im.max()
                if mx > 0: im = im / mx
                if not a.no_detector_noise:
                    im = add_detector_noise(im, noise_rng)
                views.append((im * 255).astype(np.uint8))
            arr = np.stack(views)
            if not np.isfinite(arr).all(): skipped += 1; continue
            sid = system_id(sg.number)
            if a.cap and per_sys[sid] >= a.cap:
                skipped += 1; continue
            per_sys[sid] += 1
            IMGS.append(arr); LAB.append(sid); IDS.append(cid)
        except Exception:
            skipped += 1; continue

    #Amorphous class
    if a.amorphous > 0:
        rng = np.random.default_rng(a.seed)
        for i in range(a.amorphous):
            IMGS.append(make_amorphous_views(rng, n_views=len(ORIENTATIONS),
                                             img=a.img, scale=a.scale))
            LAB.append(AMORPHOUS_LABEL); IDS.append(f"amorphous_{i:04d}")
        print(f"  Amorphous samples added: {a.amorphous}")


    if a.noise > 0:
        rng = np.random.default_rng(a.seed + 1)
        for i in range(a.noise):
            IMGS.append(make_noise_views(rng, n_views=len(ORIENTATIONS), img=a.img))
            LAB.append(NOISE_LABEL); IDS.append(f"noise_{i:04d}")
        print(f"  Noise/air samples added: {a.noise}")

    IMGS = np.stack(IMGS); LAB = np.array(LAB, dtype=np.int64)
    print(f"  Kept {len(IDS)} crystals+amorphous  (skipped {skipped} CIFs)")
    counts = np.bincount(LAB, minlength=len(SYSTEMS))
    print("  Per-class: " + ", ".join(f"{SYSTEMS[i]}={counts[i]}" for i in range(len(SYSTEMS))))

    dt = h5py.special_dtype(vlen=str)
    with h5py.File(a.out, "w") as h:
        h.create_dataset("images", data=IMGS, compression="gzip", compression_opts=4, chunks=True)
        h.create_dataset("labels", data=LAB, compression="gzip")
        di = h.create_dataset("crystal_ids", (len(IDS),), dtype=dt)
        for i, v in enumerate(IDS): di[i] = v
        h.attrs["systems"] = ",".join(SYSTEMS)
        h.attrs["n_views"] = IMGS.shape[1]; h.attrs["img"] = IMGS.shape[2]
    print(f"  Wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
