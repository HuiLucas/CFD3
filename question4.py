from pathlib import Path
import argparse

import numpy as np
from scipy.io import loadmat
from scipy.fft import ifftn


def ordered_wavenumbers(n):
    if n % 2 == 0:
        return np.concatenate((np.arange(0, n // 2), np.array([n // 2]), np.arange(-n // 2 + 1, 0)))
    return np.concatenate((np.arange(0, (n - 1) // 2 + 1), np.arange(-(n - 1) // 2, 0)))


def E_physical(real_space, domain_lengths):
    u = real_space["u"]
    v = real_space["v"]
    w = real_space["w"]

    nx, ny, nz = u.shape
    lx, ly, lz = domain_lengths

    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    domain_size = lx * ly * lz

    energy_density = 0.5 * (u**2 + v**2 + w**2)
    total_energy = np.sum(energy_density) * dx * dy * dz
    return total_energy/domain_size

def E_spectral(spectral, domain_lengths):
    uk = spectral["u"]
    vk = spectral["v"]
    wk = spectral["w"]

    energy_density_hat = 0.5 * (uk * uk.conjugate() + vk * vk.conjugate() + wk * wk.conjugate())
    E_avg = np.sum(energy_density_hat).real
    return E_avg





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

    energy_physical = E_physical(real_space, domain_lengths)
    energy_spectral = E_spectral(spectral, domain_lengths)
    print(f"Total kinetic energy from physical space: {energy_physical:.6e}")
    print(f"Total kinetic energy from spectral space: {energy_spectral:.6e}")