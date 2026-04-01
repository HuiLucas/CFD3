from pathlib import Path

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


def vorticity_z(spectral, domain_lengths):
    uk = spectral["u"]
    vk = spectral["v"]
    wk = spectral["w"]

    nx, ny, nz = uk.shape
    lx, ly, lz = domain_lengths

    n_total = nx * ny * nz
    u_hat = uk * n_total
    v_hat = vk * n_total
    w_hat = wk * n_total

    kx = fftfreq(nx, d=lx / nx) * 2 * np.pi  # angular wavenumber is convention in CFD
    ky = fftfreq(ny, d=ly / ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz / nz) * 2 * np.pi

    return ifftn(1j * (kx[:, None, None] * v_hat - ky[None, :, None] * u_hat)).real

def velocity_magnitude(spectral, domain_lengths):
    uk = spectral["u"]
    vk = spectral["v"]
    wk = spectral["w"]

    nx, ny, nz = uk.shape
    n_total = nx * ny * nz

    u, v, w = ifftn(uk*n_total).real, ifftn(vk*n_total).real, ifftn(wk*n_total).real

    return np.sqrt(u ** 2 + v ** 2 + w ** 2)



def _shared_limits(a, b):
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if np.isclose(lo, hi):
        hi = lo + 1e-12
    return lo, hi




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

    omega_dns = vorticity_z(spectral, domain_lengths)
    omega_les = vorticity_z(filtered_spectral, domain_lengths)
    vel_dns = velocity_magnitude(spectral, domain_lengths)
    vel_les = velocity_magnitude(filtered_spectral, domain_lengths)

    nx, ny, nz = shape
    z_index = nz // 2


    x = np.linspace(0.0, domain_lengths[0], nx, endpoint=False)
    y = np.linspace(0.0, domain_lengths[1], ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    omega_dns_xy = omega_dns[:, :, z_index]
    omega_les_xy = omega_les[:, :, z_index]
    vel_dns_xy = vel_dns[:, :, z_index]
    vel_les_xy = vel_les[:, :, z_index]

    wmin, wmax = _shared_limits(omega_dns_xy, omega_les_xy)
    vmin, vmax = _shared_limits(vel_dns_xy, vel_les_xy)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    c0 = axes[0, 0].contourf(xx, yy, omega_dns_xy, levels=20, cmap="RdBu_r", vmin=wmin, vmax=wmax)
    axes[0, 0].set_title(r"DNS $\omega_z$")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(c0, ax=axes[0, 0])

    c1 = axes[0, 1].contourf(xx, yy, omega_les_xy, levels=20, cmap="RdBu_r")
    axes[0, 1].set_title(r"Filtered LES $\omega_z$")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    fig.colorbar(c1, ax=axes[0, 1])

    c2 = axes[1, 0].contourf(xx, yy, vel_dns_xy, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(r"DNS $|\mathbf{u}|$")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    fig.colorbar(c2, ax=axes[1, 0])

    c3 = axes[1, 1].contourf(xx, yy, vel_les_xy, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(r"Filtered LES $|\mathbf{u}|$")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    fig.colorbar(c3, ax=axes[1, 1])

    fig.suptitle(f"Vorticity and velocity magnitude on x-y plane at z-index {z_index}")



    plt.show()
