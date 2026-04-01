from pathlib import Path
import argparse

import numpy as np
from scipy.io import loadmat
from scipy.fft import ifftn, fftfreq

import matplotlib.pyplot as plt


@np.vectorize(otypes=[np.ndarray, np.ndarray] , excluded={1,2,3})
def E_spec_3D(n, spectral, domain_lengths, k_0):
    uk = spectral["u"]
    vk = spectral["v"]
    wk = spectral["w"]

    nx, ny, nz = uk.shape
    lx, ly, lz = domain_lengths

    kx = fftfreq(nx, d=lx / nx) * 2 * np.pi  # angular wavenumber is convention in CFD
    ky = fftfreq(ny, d=ly / ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz / nz) * 2 * np.pi

    k_upper_lim = k_0 * (n + 0.5)
    if n != 0:
        k_lower_lim = k_0 * (n - 0.5)
    else:
        k_lower_lim = 0.0

    k_abs = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
    mask = (k_abs >= k_lower_lim) & (k_abs < k_upper_lim)

    energy_density_hat = 0.5 * (uk * uk.conjugate() + vk * vk.conjugate() + wk * wk.conjugate())
    E_n = np.sum(energy_density_hat[mask]).real
    return n*k_0, E_n








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
    lx = domain_lengths[0]
    k_0 = 2 * np.pi / lx
    nx = spectral["u"].shape[0]
    k_array, E_spectrum = E_spec_3D(np.arange(0, np.ceil(np.max(fftfreq(nx, d=lx/nx)*2*np.pi/k_0)*np.sqrt(3))+1) , spectral, domain_lengths, k_0)
    plt.figure()
    plt.plot(k_array, E_spectrum, marker=".", linestyle="-")
    plt.semilogx()
    plt.semilogy()
    plt.xlabel("Wavenumber k")
    plt.ylabel("Energy E(k)")
    plt.title("Energy spectrum E(k) vs wavenumber k")
    plt.grid()
    plt.show()


