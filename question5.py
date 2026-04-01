from pathlib import Path
import argparse

import numpy as np
from scipy.io import loadmat
from scipy.fft import ifftn, fftfreq


def ordered_wavenumbers(n):
    if n % 2 == 0:
        return np.concatenate((np.arange(0, n // 2), np.array([n // 2]), np.arange(-n // 2 + 1, 0)))
    return np.concatenate((np.arange(0, (n - 1) // 2 + 1), np.arange(-(n - 1) // 2, 0)))


def dissip_rate(spectral, domain_lengths):
    uk = spectral["u"]
    vk = spectral["v"]
    wk = spectral["w"]

    nx, ny, nz = uk.shape
    lx, ly, lz = domain_lengths

    kx = fftfreq(nx, d=lx / nx) * 2 * np.pi  # angular wavenumber is convention in CFD
    ky = fftfreq(ny, d=ly / ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz / nz) * 2 * np.pi

    k_squared = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    energy_density_hat = uk * uk.conjugate() + vk * vk.conjugate() + wk * wk.conjugate()
    dissipation_density_hat = k_squared * energy_density_hat
    dissipation_rate = np.sum(dissipation_density_hat).real * 0.0008
    return dissipation_rate





if __name__ == "__main__":


    base_dir = Path(__file__).resolve().parent
    fourier = loadmat(base_dir / "uvw_fourier.mat")
    physical = loadmat(base_dir / "uvw_physical.mat")

    spectral = {"u": fourier["uk"], "v": fourier["vk"], "w": fourier["wk"]}
    real_space = {"u": physical["u"], "v": physical["v"], "w": physical["w"]}
    shape = real_space["u"].shape
    if shape != spectral["u"].shape:
        raise ValueError(f"Shape mismatch: physical {shape}, spectral {spectral['u'].shape}")

    domain_lengths = (2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi)

    dissipation_rate = dissip_rate(spectral, domain_lengths)
    print(f"Dissipation rate: {dissipation_rate:.6e}")
