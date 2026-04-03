from pathlib import Path
import argparse

import numpy as np
from scipy.io import loadmat
from scipy.fft import ifftn, fftfreq

import matplotlib.pyplot as plt


def LES_filter(spectral, domain_lengths, new_N):
    uk = spectral["u"]
    vk = spectral["v"]
    wk = spectral["w"]

    nx, ny, nz = uk.shape
    lx, ly, lz = domain_lengths

    kx = fftfreq(nx, d=lx / nx) * 2 * np.pi  # angular wavenumber is convention in CFD
    ky = fftfreq(ny, d=ly / ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz / nz) * 2 * np.pi

    Nyquist = 0.5* new_N / max(lx, ly, lz) * 2 * np.pi  # convert to angular wavenumber
    k_abs = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
    mask = k_abs > Nyquist
    W = max(lx, ly, lz) / new_N
    kernel = lambda x: np.sinc(W*x/(2*np.pi))
    uk_new = uk * kernel(kx[:, None, None]) * kernel(ky[None, :, None]) * kernel(kz[None, None, :])
    uk_new[mask] = 0.0
    vk_new = vk * kernel(kx[:, None, None]) * kernel(ky[None, :, None]) * kernel(kz[None, None, :])
    vk_new[mask] = 0.0
    wk_new = wk * kernel(kx[:, None, None]) * kernel(ky[None, :, None]) * kernel(kz[None, None, :])
    wk_new[mask] = 0.0
    return {"u": uk_new, "v": vk_new, "w": wk_new}

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


@np.vectorize(otypes=[np.ndarray, np.ndarray] , excluded={1,2,3})
def diss_spec_3D(n, spectral, domain_lengths, k_0):
    k_n, E_n = E_spec_3D(n, spectral, domain_lengths, k_0)
    diss_n = 2 * (k_n**2) * E_n * 0.0008
    return k_n, diss_n


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

    new_N = 24
    filtered_spectral = LES_filter(spectral, domain_lengths, new_N)
    lx = domain_lengths[0]
    k_0 = 2 * np.pi / lx
    nx = spectral["u"].shape[0]
    nx_filtered = filtered_spectral["u"].shape[0]
    k_array2, E_spectrum = E_spec_3D(np.arange(0, np.ceil(np.max(fftfreq(nx, d=lx / nx)*2*np.pi / k_0) * np.sqrt(3)) + 1),
                                     spectral, domain_lengths, k_0)
    k_array, diss_spectrum = diss_spec_3D(np.arange(0, np.ceil(np.max(fftfreq(nx, d=lx / nx)*2*np.pi / k_0) * np.sqrt(3)) + 1),
                                          spectral, domain_lengths, k_0)
    k_array3, E_spectrum_filtered = E_spec_3D(np.arange(0, np.ceil(np.max(fftfreq(nx_filtered, d=lx / nx_filtered)*2*np.pi / k_0) * np.sqrt(3)) + 1),
                                              filtered_spectral, domain_lengths, k_0)
    k_array4, diss_spectrum_filtered = diss_spec_3D(np.arange(0, np.ceil(np.max(fftfreq(nx_filtered, d=lx / nx_filtered)*2*np.pi / k_0) * np.sqrt(3)) + 1),
                                                   filtered_spectral, domain_lengths, k_0)
    plt.figure()
    plt.plot(k_array, diss_spectrum, marker=".", linestyle="-")
    plt.plot(k_array4, diss_spectrum_filtered, marker=".", linestyle="-")
    plt.xlabel(r"Wavenumber $xi$")
    plt.ylabel(r"Dissipation spectrum")
    plt.semilogx()
    # plt.semilogy()
    plt.title(r"Dissipation spectrum vs wavenumber $xi$")
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(k_array, E_spectrum, marker=".", linestyle="-")
    plt.plot(k_array3, E_spectrum_filtered, marker=".", linestyle="-")
    plt.semilogx()
    plt.semilogy()
    plt.xlabel(r"Wavenumber $xi$")
    plt.ylabel(r"Energy $E(\xi)$")
    plt.title(r"Energy spectrum $E(\xi)$ vs wavenumber $xi$")
    plt.grid()

    plt.show()


