from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.fft import ifftn, fftfreq, fftn

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



def eddy_viscosity(filtered_spectral, domain_lengths, filter_width):
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
    mag_S = np.sqrt(2 * np.sum(S_tensor ** 2, axis=(0, 1)))

    C_s = 0.17
    delta = filter_width

    nu_t = (C_s * delta) ** 2 * mag_S

    return nu_t






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
    filter_width = np.cbrt((domain_lengths[0] * domain_lengths[1] * domain_lengths[2]) / (new_N**3))


    eddy_visc = eddy_viscosity(filtered_spectral, domain_lengths, filter_width)

    plt.figure(figsize=(8, 6))
    mid_slice = eddy_visc[:, :, eddy_visc.shape[2] // 2]
    im = plt.imshow(mid_slice, origin='lower', extent=(0, domain_lengths[0], 0, domain_lengths[1]), cmap='viridis')
    plt.colorbar(im, label='Eddy Viscosity')
    plt.title('Eddy Viscosity in Mid-Plane')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()