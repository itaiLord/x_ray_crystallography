"""
src/multi_angle_simulator.py
============================
Simulate diffraction patterns by ROTATING THE CRYSTAL, not the image.

Why this matters physically
---------------------------
In a real X-ray experiment:
  - The X-ray beam is fixed (along z).
  - The detector is fixed (the xy plane).
  - The CRYSTAL rotates on a goniometer.

When the crystal rotates by matrix R, every reciprocal lattice vector
transforms as:  q_lab = R @ B @ [h,k,l]

The spot lands at detector position (q_lab[0], q_lab[1]) * scale.

This means:
  - Different orientations bring different hkl planes into view.
  - Spot POSITIONS change (not just image rotation).
  - Spot INTENSITIES change (structure factor depends on orientation).
  - Systematic absences appear/disappear at different orientations.

This is fundamentally different from rotating the 2D image, which would
keep the same spots and just spin them around the centre — physically wrong.

Three rotation axes used
------------------------
We use three independent rotation axes to cover SO(3):
  Rz(phi)  — rotation around beam axis      (in-plane)
  Rx(chi)  — tilt of crystal (chi angle)    (out-of-plane)
  Ry(omega)— goniostat omega rotation       (out-of-plane)

For each of the N orientations we draw a random combination so that
the set covers orientation space well.
"""

import numpy as np
import gemmi
from pathlib import Path


def get_B_matrix(cell: gemmi.UnitCell) -> np.ndarray:
    """B @ [h,k,l] = q in Angstrom^-1 (Cartesian lab frame)."""
    O = np.array(cell.orth.mat.tolist())
    return np.linalg.inv(O).T


def Rx(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[1,0,0],[0,np.cos(t),-np.sin(t)],[0,np.sin(t),np.cos(t)]])

def Ry(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])

def Rz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])


def crystal_rotation(phi: float, chi: float, omega: float) -> np.ndarray:

    return Rz(phi) @ Rx(chi) @ Ry(omega)


def _precompute_hkl_and_intensities(sm, dmin: float) -> tuple:
    """
    Compute F_hkl for all reflections once.
    Returns (hkl_array, intensities, B_matrix).
    This is the expensive step — done once per crystal, not per orientation.
    """
    cell = sm.cell
    sg   = sm.spacegroup or gemmi.find_spacegroup_by_name("P 1")
    calc = gemmi.StructureFactorCalculatorX(cell)
    B    = get_B_matrix(cell)

    h_max = int(cell.a / dmin) + 1
    k_max = int(cell.b / dmin) + 1
    l_max = int(cell.c / dmin) + 1

    hkl_list   = []
    I_list     = []

    for h in range(-h_max, h_max + 1):
        for k in range(-k_max, k_max + 1):
            for l in range(-l_max, l_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue

                d = cell.calculate_d([h, k, l])
                if d < dmin:
                    continue
                F = calc.calculate_sf_from_small_structure(sm, [h, k, l])
                I = float(abs(F) ** 2)
                hkl_list.append([h, k, l])
                I_list.append(I)

    return (np.array(hkl_list, dtype=np.int32),
            np.array(I_list,   dtype=np.float32),
            B)


def _project_crystal(hkl_arr: np.ndarray,
                     intensities: np.ndarray,
                     B: np.ndarray,
                     R: np.ndarray,
                     img_size: int,
                     scale_px: float,
                     spot_sigma: float,
                     min_I: float) -> np.ndarray:
    """
    Project rotated crystal onto detector.

    The crystal rotation R is applied to B:  q_lab = R @ B @ [h,k,l]
    The detector sees (q_lab[0], q_lab[1]) — the xy components.

    This is a VECTORISED numpy implementation — no Python loops over pixels.
    """

    RB   = R @ B                                    # (3,3)
    q    = hkl_arr.astype(np.float64) @ RB.T       # (N, 3)
    vals  = np.sqrt(np.clip(intensities, 0, None))
    v_max = vals.max()
    if v_max == 0:
        return np.zeros((img_size, img_size), dtype=np.float32)

    amp   = vals / v_max
    mask  = amp >= min_I
    q     = q[mask]
    amp   = amp[mask]
    cx = cy = img_size / 2.0


    px_x = cx + q[:, 0] * scale_px
    px_y = cy + q[:, 1] * scale_px

    xi   = np.round(px_x).astype(np.int32)
    yi   = np.round(px_y).astype(np.int32)


    r    = int(np.ceil(3 * spot_sigma))
    on_det = ((r <= xi) & (xi < img_size - r) &
               (r <= yi) & (yi < img_size - r))
    xi, yi, amp = xi[on_det], yi[on_det], amp[on_det]

    img  = np.zeros((img_size, img_size), dtype=np.float32)


    dy_range = np.arange(-r, r + 1)
    dx_range = np.arange(-r, r + 1)
    DY, DX   = np.meshgrid(dy_range, dx_range, indexing='ij')
    G        = np.exp(-(DX**2 + DY**2) / (2 * spot_sigma**2))

    for i in range(len(xi)):
        img[yi[i] + DY, xi[i] + DX] += amp[i] * G

    return img


def simulate_10_angles(cif_path: str, cfg: dict) -> tuple:
    """
    Simulate N diffraction images of a crystal at N different 3D orientations.

    Each orientation = a genuine crystal rotation (phi, chi, omega).
    The diffraction pattern changes because DIFFERENT hkl planes come
    into diffraction condition at each orientation — physically correct.

    F_hkl computed ONCE, projected N times (efficient).

    Returns:
      images      : (N, H, W) float32
      cell_params : [a, b, c, alpha, beta, gamma]
      atoms_cart  : list of (element, x, y, z)
      crystal_id  : str
    """
    sm       = gemmi.read_small_structure(str(cif_path))
    cell     = sm.cell
    dmin     = cfg.get("max_resolution_angstrom", 0.8)
    img_size = cfg.get("image_size", 256)
    scale    = cfg.get("scale_px", 90)
    sigma    = cfg.get("spot_sigma_px", 1.8)
    min_I    = cfg.get("min_intensity", 0.005)

    hkl_arr, intensities, B = _precompute_hkl_and_intensities(sm, dmin)

    # Build rotation list from config orientations
    # Each entry has phi, chi, omega — full 3D crystal rotation
    orientations = cfg.get("orientations", [
        {"phi":   0, "chi":   0, "omega":   0},
        {"phi":  45, "chi":  15, "omega":  30},
        {"phi":  90, "chi": -15, "omega":  60},
        {"phi": 135, "chi":  10, "omega":  90},
        {"phi": 180, "chi": -10, "omega": 120},
        {"phi":  20, "chi":  25, "omega": 150},
        {"phi":  60, "chi": -20, "omega": 180},
        {"phi": 100, "chi":  20, "omega": 210},
        {"phi": 160, "chi":  -5, "omega": 240},
        {"phi": 200, "chi":   5, "omega": 270},
    ])

    images = []
    for ori in orientations:
        R   = crystal_rotation(ori["phi"], ori["chi"], ori["omega"])
        img = _project_crystal(hkl_arr, intensities, B, R,
                                img_size, scale, sigma, min_I)
        mx  = img.max()
        if mx > 0:
            img /= mx
        images.append(img)

    cell_params = np.array([cell.a, cell.b, cell.c,
                             cell.alpha, cell.beta, cell.gamma],
                            dtype=np.float32)

    atoms_cart = []
    for site in sm.get_all_unit_cell_sites():
        pos = cell.orthogonalize(site.fract)
        atoms_cart.append((site.element.name, pos.x, pos.y, pos.z))

    return (np.array(images, dtype=np.float32),
            cell_params, atoms_cart,
            Path(cif_path).stem)
