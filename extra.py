from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.fft import ifftn, fftfreq, fftn

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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
    return {"u": uk_new, "v": vk_new, "w": wk_new}, W


def SGS_eddy_viscosity_deviatoric(filtered_spectral, domain_lengths, filter_width):
    lx, ly, lz = domain_lengths
    uk_f = filtered_spectral["u"]
    vk_f = filtered_spectral["v"]
    wk_f = filtered_spectral["w"]
    nx, ny, nz = uk_f.shape
    n_total_f = nx * ny * nz
    uvwk_f = np.array([uk_f, vk_f, wk_f]) * n_total_f

    kx = fftfreq(nx, d=lx / nx) * 2 * np.pi  # angular wavenumber is convention in CFD
    ky = fftfreq(ny, d=ly / ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz / nz) * 2 * np.pi

    grad_operator = np.stack((
        np.broadcast_to(kx[:, np.newaxis, np.newaxis], (nx, ny, nz)),
        np.broadcast_to(ky[np.newaxis, :, np.newaxis], (nx, ny, nz)),
        np.broadcast_to(kz[np.newaxis, np.newaxis, :], (nx, ny, nz))
    ))

    S_tensor_spec = 0.5j * (
            uvwk_f[:, None, :, :, :] * grad_operator[None, :, :, :, :] +
            grad_operator[:, None, :, :, :] * uvwk_f[None, :, :, :, :]
    )
    S_tensor = ifftn(S_tensor_spec, axes=(2, 3, 4)).real
    mag_S = np.sqrt(2*np.sum(S_tensor**2, axis=(0,1)))

    C_s = 0.17
    delta = filter_width

    nu_t = (C_s * delta)**2 * mag_S

    return -2*nu_t[None, None, :, :, :] * S_tensor



def _shared_limits(a, b):
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if np.isclose(lo, hi):
        hi = lo + 1e-12
    return lo, hi


def compute_filtered_strain_rate(filtered_spectral, domain_lengths):
    """Calculates the resolved strain rate tensor S_ij in physical space."""
    lx, ly, lz = domain_lengths
    uk_f = filtered_spectral["u"]
    vk_f = filtered_spectral["v"]
    wk_f = filtered_spectral["w"]
    nx, ny, nz = uk_f.shape
    n_total_f = nx * ny * nz
    uvwk_f = np.array([uk_f, vk_f, wk_f]) * n_total_f

    kx = fftfreq(nx, d=lx / nx) * 2 * np.pi
    ky = fftfreq(ny, d=ly / ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz / nz) * 2 * np.pi

    grad_operator = np.stack((
        np.broadcast_to(kx[:, np.newaxis, np.newaxis], (nx, ny, nz)),
        np.broadcast_to(ky[np.newaxis, :, np.newaxis], (nx, ny, nz)),
        np.broadcast_to(kz[np.newaxis, np.newaxis, :], (nx, ny, nz))
    ))

    # Calculate complex strain rate in spectral space
    S_tensor_spec = 0.5j * (
            uvwk_f[:, None, :, :, :] * grad_operator[None, :, :, :, :] +
            grad_operator[:, None, :, :, :] * uvwk_f[None, :, :, :, :]
    )

    # Return physical space strain rate tensor
    return ifftn(S_tensor_spec, axes=(2, 3, 4)).real


def calculate_dissipation(tau, S):
    """Calculates the SGS energy dissipation rate: -tau_ij * S_ij"""
    # np.einsum efficiently computes the double dot product over the spatial grid
    return -np.einsum('ij...,ij...->...', tau, S)


def SGS_deviatoric(spectral, domain_lengths, new_N):
    lx, ly, lz = domain_lengths

    uk = spectral["u"]
    vk = spectral["v"]
    wk = spectral["w"]
    nx, ny, nz = uk.shape
    n_total = nx * ny * nz
    uvwk = np.array([uk, vk, wk]) * n_total

    filtered_spectral, filter_width = LES_filter(spectral, domain_lengths, new_N)
    uk_f = filtered_spectral["u"]
    vk_f = filtered_spectral["v"]
    wk_f = filtered_spectral["w"]
    nx, ny, nz = uk_f.shape
    n_total_f = nx * ny * nz
    uvwk_f = np.array([uk_f, vk_f, wk_f]) * n_total_f

    uvw = ifftn(uvwk).real
    uvw_f = ifftn(uvwk_f).real
    first_term_unfiltered = np.einsum('i... , j... -> ij...', uvw, uvw)
    first_term_unfilteredk = fftn(first_term_unfiltered, axes=(2, 3, 4)) / n_total

    kx = fftfreq(nx, d=lx / nx) * 2 * np.pi  # angular wavenumber is convention in CFD
    ky = fftfreq(ny, d=ly / ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz / nz) * 2 * np.pi

    Nyquist = 0.5* new_N / max(lx, ly, lz) * 2 * np.pi  # convert to angular wavenumber
    k_abs = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    mask = k_abs > Nyquist
    W = max(lx, ly, lz) / new_N
    kernel = lambda x: np.sinc(W*x/(2*np.pi))
    discrete_kernel = kernel(kx[:, None, None]) * kernel(ky[None, :, None]) * kernel(kz[None, None, :])
    first_term_filteredk =  first_term_unfilteredk * discrete_kernel[None, None, :, :, :] * n_total
    first_term_filteredk[..., mask] = 0.0

    first_term_filtered = ifftn(first_term_filteredk, axes=(2, 3, 4)).real

    second_term = np.einsum('i... , j... -> ij...', uvw_f, uvw_f)
    SGS = first_term_filtered - second_term
    return  SGS - np.eye(3)[..., None, None, None] * np.trace(SGS, axis1=0, axis2=1)[None, None, :, :, :] / 3.0



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

    # 1. Get filtered fields and basic properties
    filtered_spectral, filter_width = LES_filter(spectral, domain_lengths, new_N)
    nx, ny, nz = shape
    z_index = nz // 2

    # Assuming your Smagorinsky function returns the tensor and filter_width is defined
    W = max(domain_lengths) / new_N

    # 2. Calculate the Strain Rate Tensor S_ij
    S_tensor = compute_filtered_strain_rate(filtered_spectral, domain_lengths)

    # 3. Calculate both SGS Tensors
    tau_exact = SGS_deviatoric(spectral, domain_lengths, new_N)

    # (Make sure SGS_eddy_viscosity_deviatoric is using the updated factor of 2!)
    tau_smag = SGS_eddy_viscosity_deviatoric(filtered_spectral, domain_lengths, W)

    # 4. Calculate Dissipation Fields
    dissipation_exact = calculate_dissipation(tau_exact, S_tensor)
    dissipation_smag = calculate_dissipation(tau_smag, S_tensor)

    # 5. Extract the 2D slices for plotting
    diss_exact_xy = dissipation_exact[:, :, z_index]
    diss_smag_xy = dissipation_smag[:, :, z_index]

    # --- Plotting ---
    x = np.linspace(0.0, domain_lengths[0], nx, endpoint=False)
    y = np.linspace(0.0, domain_lengths[1], ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Exact Dissipation
    lim_exact = max(abs(np.min(diss_exact_xy)), abs(np.max(diss_exact_xy)))
    norm_exact = TwoSlopeNorm(vmin=-lim_exact, vcenter=0., vmax=lim_exact)
    c0 = axes[0].contourf(xx, yy, diss_exact_xy, levels=21, cmap="RdBu_r", norm=norm_exact)
    axes[0].set_title(r"Exact SGS Dissipation ($\epsilon_{sgs}$)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(c0, ax=axes[0])

    # Smagorinsky Dissipation
    # Note: Smagorinsky is strictly dissipative (positive), so we can use a sequential colormap starting at 0
    c1 = axes[1].contourf(xx, yy, diss_smag_xy, levels=21, cmap="Reds", vmin=0.0)
    axes[1].set_title(r"Smagorinsky SGS Dissipation ($\epsilon_{sgs}$)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(c1, ax=axes[1])

    fig.suptitle(f"SGS Energy Dissipation Comparison at z-index {z_index}")
    plt.show()