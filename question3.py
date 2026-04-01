from pathlib import Path
import argparse

import numpy as np
from scipy.io import loadmat
from scipy.fft import ifftn, fftfreq



def Q_criterion(spectral, domain_lengths):
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

    vel_hat = np.array([u_hat, v_hat, w_hat])
    grad_hat = np.stack((
        1j * kx[:, None, None] * vel_hat,
        1j * ky[None, :, None] * vel_hat,
        1j * kz[None, None, :] * vel_hat,
    ))

    # Transform only spatial axes; grad_hat has leading tensor axes.
    grad_phys = ifftn(grad_hat, axes=(2, 3, 4)).real
    q_criterion = -0.5 * np.sum(grad_phys * grad_phys.transpose((1, 0, 2, 3, 4)), axis=(0, 1))
    return q_criterion


def numpy_to_pyvista_image_data(scalars, domain_lengths, scalar_name="Q"):
    import pyvista as pv

    nx, ny, nz = scalars.shape
    lx, ly, lz = domain_lengths

    # ImageData stores point-centered arrays on a regular Cartesian lattice.
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(lx / nx, ly / ny, lz / nz),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data[scalar_name] = np.asarray(scalars, dtype=np.float32).ravel(order="F")
    return grid



def render_contour(contour, window_title="Q-criterion iso-surface"):
    import pyvista as pv

    plotter = pv.Plotter(window_size=(1100, 800), title=window_title)
    plotter.set_background("#ffffff")
    plotter.add_mesh(contour, smooth_shading=True)
    plotter.show_axes()
    plotter.show()


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
    q_criterion = Q_criterion(spectral, domain_lengths)


    iso_value = np.percentile(q_criterion, 5)
    iso_value2 = np.percentile(q_criterion, 30)
    iso_value3 = np.percentile(q_criterion, 80)


    image_data = numpy_to_pyvista_image_data(q_criterion, domain_lengths)
    contour = image_data.contour(isosurfaces=[float(iso_value), float(iso_value2), float(iso_value3)], scalars='Q')
    render_contour(contour)
