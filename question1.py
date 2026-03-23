from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.fft import fftn, ifftn


def verify_fft_pairs(spectral, real_space):
    results = {}

    for comp in ("u", "v", "w"):
        u = real_space[comp]
        uk = spectral[comp]
        n_total = u.size

        uk_from_u = fftn(u, s = (192, 192, 192)) / n_total
        u_from_uk = ifftn(uk * n_total, s = (192, 192, 192))

        results[comp] = {
            "forward_rel_l2": np.linalg.norm((uk_from_u - uk).ravel()) / np.linalg.norm(uk.ravel()),
            "inverse_rel_l2": np.linalg.norm((u_from_uk - u).ravel()) / np.linalg.norm(u.ravel()),
        }

    return results


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    fourier = loadmat(base_dir / "uvw_fourier.mat")
    physical = loadmat(base_dir / "uvw_physical.mat")

    spectral = {"u": fourier["uk"], "v": fourier["vk"], "w": fourier["wk"]}
    real_space = {"u": physical["u"], "v": physical["v"], "w": physical["w"]}
    shape = real_space["u"].shape
    if shape != spectral["u"].shape:
        raise ValueError(f"Shape mismatch: physical {shape}, spectral {spectral['u'].shape}")

    nx, ny, nz = shape

    results = verify_fft_pairs(spectral, real_space)

    print("3D FFT verification for uvw data")
    print(f"grid shape: {shape}, total points: {nx * ny * nz}")
    for comp in ("u", "v", "w"):
        r = results[comp]
        print(
            f"  {comp}: "
            f"forward relL2={r['forward_rel_l2']:.3e}, "
            f"inverse relL2={r['inverse_rel_l2']:.3e}, "

        )
