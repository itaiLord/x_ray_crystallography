"""
src/download_cod.py
===================
Download REAL crystal structures from the Crystallography Open Database (COD)
into data/cif_cache/, ready for src/regenerate_xrd2d.py.

Two modes:

1) Named curated crystals (saved as <RealName>.cif, e.g. NaCl.cif):
     python src/download_cod.py
     python src/download_cod.py --name CaCO3 --term calcite
     python src/download_cod.py --name MyXtal --id 1000041

2) Bulk by CRYSTAL SYSTEM — to BALANCE under-represented classes (e.g. triclinic):
     python src/download_cod.py --system triclinic --count 350
     python src/download_cod.py --system monoclinic --count 200
   Files are saved as <formula>_cod<id>.cif (formula read from the CIF).

Needs internet (COD is public). If your network blocks www.crystallography.net
you'll see connection errors — run it where the internet is open, or download
CIFs by hand from the COD website into data/cif_cache/.
"""
import os, sys, time, argparse
from collections import Counter
from math import gcd
from functools import reduce
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import requests, gemmi
from pathlib import Path

COD_RESULT = "https://www.crystallography.net/cod/result"
COD_CIF    = "https://www.crystallography.net/cod/{cod_id}.cif"

CURATED = {
    "NaCl":"halite","KCl":"sylvite","CaF2":"fluorite","MgO":"periclase",
    "FeS2_pyrite":"pyrite","ZnS_sphalerite":"sphalerite","SiO2_quartz":"quartz",
    "CaCO3_calcite":"calcite","Al2O3_corundum":"corundum","TiO2_rutile":"rutile",
    "TiO2_anatase":"anatase","CaTiO3":"perovskite","S8_sulfur":"sulphur",
    "CaSO4_2H2O":"gypsum","KAlSi3O8":"orthoclase","CuSO4_5H2O":"chalcanthite",
    "Al2SiO5_kyanite":"kyanite","Mg3Si4O10_talc":"talc","C_graphite":"graphite",
    "ZnO":"zincite",
}

# Crystal system -> representative space-group numbers to query.
# Triclinic is small (1,2); for higher-symmetry systems we sample several groups.
SYSTEM_SPACEGROUPS = {
    "triclinic":    [1, 2],
    "monoclinic":   [4, 11, 14, 15],
    "orthorhombic": [19, 33, 62, 63, 64],
    "tetragonal":   [87, 136, 139, 141],
    "trigonal":     [148, 161, 166, 167],
    "hexagonal":    [176, 186, 194],
    "cubic":        [216, 221, 225, 227, 229],
}
SYS = ["triclinic","monoclinic","orthorhombic","tetragonal","trigonal","hexagonal","cubic"]
def _system(num):
    return SYS[(0 if num<=2 else 1 if num<=15 else 2 if num<=74 else 3 if num<=142
               else 4 if num<=167 else 5 if num<=194 else 6)]


def _formula(sm):
    """Reduced chemical formula string from a gemmi small structure, e.g. 'CuSO4'."""
    try:
        els = Counter(s.element.name for s in sm.get_all_unit_cell_sites())
        if not els:
            els = Counter(s.element.name for s in sm.sites)
        counts = list(els.values())
        g = reduce(gcd, counts) if counts else 1
        parts = []
        for el, n in sorted(els.items()):
            n //= max(g, 1)
            parts.append(el + (str(n) if n > 1 else ""))
        return "".join(parts) or "unknown"
    except Exception:
        return "unknown"


def search_text(term, n_max=5):
    try:
        r = requests.get(COD_RESULT, params={"text": term, "format": "lst"}, timeout=25)
        return [ln.strip() for ln in r.text.splitlines() if ln.strip().isdigit()][:n_max] if r.ok else []
    except Exception as e:
        print(f"    [search '{term}'] {e}"); return []


def search_spacegroup(sg, n_max=400):
    try:
        r = requests.get(COD_RESULT, params={"space_group_number": sg, "format": "lst"}, timeout=30)
        return [ln.strip() for ln in r.text.splitlines() if ln.strip().isdigit()][:n_max] if r.ok else []
    except Exception as e:
        print(f"    [sg {sg}] {e}"); return []


def fetch_cif_text(cod_id):
    try:
        r = requests.get(COD_CIF.format(cod_id=cod_id), timeout=25)
        return r.text if r.ok else None
    except Exception:
        return None


def _validate(path, max_edge):
    sm = gemmi.read_small_structure(str(path)); c = sm.cell
    if sm.spacegroup is None or min(c.a, c.b, c.c) <= 0 or len(sm.sites) == 0:
        return None
    if max(c.a, c.b, c.c) > max_edge:
        return None
    return sm


def download_named(name, cod_id, out_dir):
    dest = Path(out_dir) / f"{name}.cif"
    txt = fetch_cif_text(cod_id)
    if not txt:
        print(f"  {name:20s} COD {cod_id}: download failed"); return False
    dest.write_text(txt, encoding="utf-8")
    sm = _validate(dest, 1e9)
    if sm is None:
        dest.unlink(missing_ok=True); print(f"  {name:20s} COD {cod_id}: invalid"); return False
    print(f"  {name:20s} <- COD {cod_id}  sg#{sm.spacegroup.number:<3d} {_system(sm.spacegroup.number)}")
    return True


def download_system(system, count, out_dir, max_edge=20.0):
    """Download up to `count` NEW valid structures of a crystal system, named <formula>_cod<id>.cif."""
    sgs = SYSTEM_SPACEGROUPS[system]
    print(f"=== {system}: collecting candidates from space groups {sgs} ===")
    cands = []
    for sg in sgs:
        cands += search_spacegroup(sg, n_max=max(count * 3, 200))
    seen = set(); cands = [c for c in cands if not (c in seen or seen.add(c))]
    import random; random.shuffle(cands)
    print(f"  {len(cands)} candidate IDs; downloading until {count} valid ...")
    got = 0
    for cid in cands:
        if got >= count:
            break
        tmp = Path(out_dir) / f"_cod{cid}.cif"
        txt = fetch_cif_text(cid)
        if not txt:
            continue
        tmp.write_text(txt, encoding="utf-8")
        sm = _validate(tmp, max_edge)
        if sm is None:
            tmp.unlink(missing_ok=True); continue
        name = f"{_formula(sm)}_cod{cid}"
        dest = Path(out_dir) / f"{name}.cif"
        if dest.exists():
            tmp.unlink(missing_ok=True); continue
        tmp.rename(dest)
        got += 1
        if got % 25 == 0:
            print(f"    ... {got}/{count}")
        time.sleep(0.15)
    print(f"  Done: {got} new {system} structures saved to {out_dir}")
    print(f"  Now rebuild: python src/regenerate_xrd2d.py")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "cif_cache"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--term", default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--system", choices=list(SYSTEM_SPACEGROUPS), default=None,
                    help="bulk-download a crystal system to balance classes")
    ap.add_argument("--count", type=int, default=300, help="how many to fetch with --system")
    ap.add_argument("--max-edge", type=float, default=20.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.system:
        download_system(a.system, a.count, a.out, a.max_edge); return
    if a.name and (a.term or a.id):
        cod_id = a.id or (search_text(a.term, 1) or [None])[0]
        if not cod_id: print(f"  no COD match for '{a.term}'"); return
        download_named(a.name, cod_id, a.out); return

    print(f"=== Downloading {len(CURATED)} named crystals from COD -> {a.out} ===")
    ok = 0
    for name, term in CURATED.items():
        if (Path(a.out) / f"{name}.cif").exists():
            print(f"  {name:20s} already present"); ok += 1; continue
        for cid in search_text(term, n_max=5):
            if download_named(name, cid, a.out): ok += 1; break
        time.sleep(0.2)
    print(f"\n  Kept/downloaded {ok}/{len(CURATED)}.")


if __name__ == "__main__":
    main()
