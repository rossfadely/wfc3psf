import os
import numpy as np

from scipy.interpolate import RectBivariateSpline

def render_psfs(psf_model, shifts, patch_shape, psf_grid, k):
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

    # NOTE - transpose of interp evaluation.
    for i in range(psfs.shape[0]):
        psfs[i] = interp_func(x + shifts[i, 0], y + shifts[i, 1]).T.ravel()

    return psfs

def generate_fake(N, psf, floor=0.05, gain=0.01, bkgrng=(10., 100.),
                  fluxrng=(1000., 10000.), pshape=(25, 25), seed=12435):
    """
    Make a bunch of (simple) patches from a psf.
    """
    shifts = np.random.rand(N, 2)
    fluxes = np.random.rand(N) * (fluxrng[1] - fluxrng[0]) + fluxrng[0]
    bkgs = np.random.rand(N) * (bkgrng[1] - bkgrng[0]) + bkgrng[0]

    #psfs = render_psfs(psf, shifts, pshape, grid, 1)

if __name__ == '__main__':

    psfdir = os.environ['wfc3psfs']
    f = pf.open(psfdir + 'tinytim-pixelconvolved-507-507-25-25-101-101.fits')
    psf = f[1].data
    f.close()

    N = 500
    generate_fake(N, psf)
