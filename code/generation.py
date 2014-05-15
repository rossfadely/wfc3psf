import numpy as np

from scipy.special import erf
from scipy.interpolate import RectBivariateSpline

def render_psfs(psf_model, shifts, patch_shape, psf_grid, k=3):
    """
    Make psfs under current model.
    """
    interp_func = RectBivariateSpline(psf_grid[0], psf_grid[1], psf_model,
                                      kx=k, ky=k)
    psfs = np.zeros((shifts.shape[0], patch_shape[0] * patch_shape[1]))

    # patch shape is odd, should be checked elsewhere...
    dx = (patch_shape[0] - 1) / 2.
    dy = (patch_shape[1] - 1) / 2.
    x = np.linspace(-dx, dx, patch_shape[0])
    y = np.linspace(-dy, dy, patch_shape[1])

    # NOTE - transpose of interp eval.
    for i in range(psfs.shape[0]):
        psfs[i] = interp_func(x + shifts[i, 0], y + shifts[i, 1]).T.ravel()

    return psfs
