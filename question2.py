from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.fft import fftn, ifftn, fftfreq

rms = lambda x: np.sqrt(np.mean(x**2))


def divergence_rms_spectral(spectral, domain_lengths):
    uk = spectral["u"]
    vk = spectral["v"]
    wk = spectral["w"]

    nx, ny, nz = uk.shape
    lx, ly, lz = domain_lengths

    n_total = nx * ny * nz
    u_hat = uk * n_total
    v_hat = vk * n_total
    w_hat = wk * n_total

    kx = fftfreq(nx, d=lx/nx) * 2 * np.pi # angular wavenumber is convention in CFD
    ky = fftfreq(ny, d=ly/ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz/nz) * 2 * np.pi

    div_hat = (
        1j * kx[:, None, None] * u_hat
        + 1j * ky[None, :, None] * v_hat
        + 1j * kz[None, None, :] * w_hat
    )
    div = ifftn(div_hat)

    div_real = div.real
    return rms(div_real), float(np.max(np.abs(div.imag))), div_real


def divergence_rms_fd2(real_space, domain_lengths):
    u = real_space["u"]
    v = real_space["v"]
    w = real_space["w"]

    nx, ny, nz = u.shape
    lx, ly, lz = domain_lengths
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz


    # periodic boundary conditions -> np.roll is possible
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dy)
    dwdz = (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2.0 * dz)

    div = dudx + dvdy + dwdz
    return rms(div), div


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

    domain_lengths = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    rms_fd2_div, fd2_div = divergence_rms_fd2(real_space, domain_lengths)
    rms_spectral, max_imag_spectral, div_spectral = divergence_rms_spectral(spectral, domain_lengths)

    print(f"RMS divergence (FD2): {rms_fd2_div:.3e}")
    print(f"RMS divergence (Spectral): {rms_spectral:.3e}, max imag part: {max_imag_spectral:.3e}")
    print(fd2_div.mean(), div_spectral.mean())


