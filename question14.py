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

def Bardina_deviatoric(filtered_spectral, domain_lengths, filter_width):
    lx, ly, lz = domain_lengths
    uk_f = filtered_spectral["u"]
    vk_f = filtered_spectral["v"]
    wk_f = filtered_spectral["w"]
    nx, ny, nz = uk_f.shape
    n_total_f = nx * ny * nz
    uvwk_f = np.array([uk_f, vk_f, wk_f]) * n_total_f

    uvw_f = ifftn(uvwk_f, axes=(1,2,3)).real
    Leonard_stress_prod_first_term = np.einsum('i... , j... -> ij...', uvw_f, uvw_f)
    Leonard_stress_prod_first_termk = fftn(Leonard_stress_prod_first_term, axes=(2, 3, 4)) / n_total_f
    kx = fftfreq(nx, d=lx / nx) * 2 * np.pi  # angular wavenumber is convention in CFD
    ky = fftfreq(ny, d=ly / ny) * 2 * np.pi
    kz = fftfreq(nz, d=lz / nz) * 2 * np.pi

    Nyquist = 0.5* new_N / max(lx, ly, lz) * 2 * np.pi  # convert to angular wavenumber
    k_abs = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    mask = k_abs > Nyquist
    W = max(lx, ly, lz) / new_N
    kernel = lambda x: np.sinc(W * x / (2 * np.pi))
    discrete_kernel = kernel(kx[:, None, None]) * kernel(ky[None, :, None]) * kernel(kz[None, None, :])
    Leonard_stress_first_term_filteredk = Leonard_stress_prod_first_termk * discrete_kernel[None, None, :, :, :] * n_total_f
    Leonard_stress_first_term_filteredk[..., mask] = 0.0
    Leonard_stress_first_term_filtered = ifftn(Leonard_stress_first_term_filteredk, axes=(2, 3, 4)).real
    Leonard_stress_second_term_prod = Leonard_stress_prod_first_term
    Leonard_stress = Leonard_stress_first_term_filtered - Leonard_stress_second_term_prod

    Twice_filtered = uvwk_f * discrete_kernel[None, :, :, :]
    Twice_filtered[..., mask] = 0.0
    Twice_filtered = ifftn(Twice_filtered, axes=(1, 2, 3)).real
    Reynolds_stress = np.einsum('i... , j... -> ij...', uvw_f - Twice_filtered, uvw_f - Twice_filtered)
    Cross_stress_first_term = np.einsum('i... , j... -> ij...', Twice_filtered, uvw_f - Twice_filtered)
    Cross_stress_second_term = np.einsum('i... , j... -> ij...', uvw_f - Twice_filtered, Twice_filtered)
    Cross_stress = Cross_stress_first_term + Cross_stress_second_term

    C_B = 1

    Mij = (Cross_stress + Reynolds_stress) * C_B

    SGS = Leonard_stress + Mij
    return SGS - np.trace(SGS, axis1=0, axis2=1)[None, None, :, :, :]/3.0


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

    uvw = ifftn(uvwk, axes=(1,2,3)).real
    uvw_f = ifftn(uvwk_f, axes=(1,2,3)).real
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
    filtered_spectral, filter_width = LES_filter(spectral, domain_lengths, new_N)

    SGS_tensor =     SGS_eddy_viscosity_deviatoric(filtered_spectral, domain_lengths, filter_width)
    nx, ny, nz = shape
    z_index = nz // 2

    Bardina_SGS_deviatoric = Bardina_deviatoric(filtered_spectral, domain_lengths, filter_width)
    SGS_tensor2 = SGS_deviatoric(spectral, domain_lengths, new_N)

    covariance = np.corrcoef(SGS_tensor2.flatten(), SGS_tensor.flatten())
    print(covariance)


    x = np.linspace(0.0, domain_lengths[0], nx, endpoint=False)
    y = np.linspace(0.0, domain_lengths[1], ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    SGS_xy = SGS_tensor[:, :, :, :, z_index]

    minn, maxx = _shared_limits(SGS_xy, SGS_xy)
    maxxx = max(abs(minn), abs(maxx))
    minn, maxx = -maxxx, maxxx
    norm = TwoSlopeNorm(vmin=minn, vcenter=0., vmax=maxx)
    levels = np.linspace(minn, maxx, 21)
    fig, axes = plt.subplots(2, 3, figsize=(15, 12), constrained_layout=True)

    c0 = axes[0, 0].contourf(xx, yy, SGS_xy[0, 0, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[0, 0].set_title(r"SGS $\tau_{0,0}$")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(c0, ax=axes[0, 0])

    c1 = axes[0, 1].contourf(xx, yy, SGS_xy[0, 1, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[0, 1].set_title(r"SGS $\tau_{0,1}$")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    fig.colorbar(c1, ax=axes[0, 1])

    c2 = axes[0, 2].contourf(xx, yy, SGS_xy[0, 2, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[0, 2].set_title(r"SGS $\tau_{0,2}$")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("y")
    fig.colorbar(c2, ax=axes[0, 2])

    c4 = axes[1, 1].contourf(xx, yy, SGS_xy[1, 1, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[1, 1].set_title(r"SGS $\tau_{1,1}$")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    fig.colorbar(c4, ax=axes[1, 1])

    c5 = axes[1, 2].contourf(xx, yy, SGS_xy[1, 2, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[1, 2].set_title(r"SGS $\tau_{1,2}$")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    fig.colorbar(c5, ax=axes[1, 2])

    c8 = axes[1, 0].contourf(xx, yy, SGS_xy[2, 2, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[1, 0].set_title(r"SGS $\tau_{2,2}$")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    fig.colorbar(c8, ax=axes[1, 0])

    fig.suptitle(f"SGS tensor components on x-y plane at z-index {z_index} for the Smagorinsky model")

    plt.show()

    x = np.linspace(0.0, domain_lengths[0], nx, endpoint=False)
    y = np.linspace(0.0, domain_lengths[1], ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    SGS_xy = Bardina_SGS_deviatoric[:, :, :, :, z_index]

    minn, maxx = _shared_limits(SGS_xy, SGS_xy)
    maxxx = max(abs(minn), abs(maxx))
    minn, maxx = -maxxx, maxxx
    norm = TwoSlopeNorm(vmin=minn, vcenter=0., vmax=maxx)
    levels = np.linspace(minn, maxx, 21)
    fig, axes = plt.subplots(2, 3, figsize=(15, 12), constrained_layout=True)

    c0 = axes[0, 0].contourf(xx, yy, SGS_xy[0, 0, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[0, 0].set_title(r"SGS $\tau_{0,0}$")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(c0, ax=axes[0, 0])

    c1 = axes[0, 1].contourf(xx, yy, SGS_xy[0, 1, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[0, 1].set_title(r"SGS $\tau_{0,1}$")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    fig.colorbar(c1, ax=axes[0, 1])

    c2 = axes[0, 2].contourf(xx, yy, SGS_xy[0, 2, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[0, 2].set_title(r"SGS $\tau_{0,2}$")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("y")
    fig.colorbar(c2, ax=axes[0, 2])

    c4 = axes[1, 1].contourf(xx, yy, SGS_xy[1, 1, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[1, 1].set_title(r"SGS $\tau_{1,1}$")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    fig.colorbar(c4, ax=axes[1, 1])

    c5 = axes[1, 2].contourf(xx, yy, SGS_xy[1, 2, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[1, 2].set_title(r"SGS $\tau_{1,2}$")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    fig.colorbar(c5, ax=axes[1, 2])

    c8 = axes[1, 0].contourf(xx, yy, SGS_xy[2, 2, :, :], levels=levels, cmap="RdBu_r", vmin=minn, vmax=maxx, norm=norm)
    axes[1, 0].set_title(r"SGS $\tau_{2,2}$")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    fig.colorbar(c8, ax=axes[1, 0])

    fig.suptitle(f"SGS tensor components on x-y plane at z-index {z_index} for the Bardina model")

    plt.show()


