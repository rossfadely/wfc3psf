import os
import numpy as np
import pyfits as pf

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

def generate_fake(N, psf, floor=0.01, gain=0.01, bkgrng=(0., 0.5),
                  fluxrng=(1.e4, 1.e5), pshape=(25, 25), seed=12435):
    """
    Make a bunch of (simple) patches from a psf.
    """
    psf_shape = psf.shape

    np.random.seed(seed)
    shifts = np.random.rand(N, 2) - 0.5
    fluxes = np.random.rand(N) * (fluxrng[1] - fluxrng[0]) + fluxrng[0]
    bkgs = np.random.rand(N) * (bkgrng[1] - bkgrng[0]) + bkgrng[0]

    xg = np.linspace(-0.5 * pshape[0], 0.5 * pshape[0],
                      psf_shape[0])
    yg = np.linspace(-0.5 * pshape[1], 0.5 * pshape[1],
                      psf_shape[1])
    grid = (xg, yg)
    psfs = render_psfs(psf, shifts, pshape, grid, 1)

    patches = bkgs[:, None] + fluxes[:, None] * psfs
    noise_sigmas = floor + gain * patches
    noise = np.random.randn(N, psfs.shape[1]) * noise_sigmas

    return shifts, bkgs, fluxes, patches, noise

if __name__ == '__main__':

    from matplotlib import use; use('Agg')
    import pyfits as pf
    import matplotlib.pyplot as pl
    from matplotlib.colors import LogNorm

    plotdir = os.environ['wfc3plots']
    psfdir = os.environ['wfc3psfs']
    f = pf.open(psfdir + 'tinytim-pixelconvolved-507-507-25-25-101-101.fits')
    psf = f[0].data
    f.close()

    N = 500
    seed = 1232
    s, b, f, p, n = generate_fake(N, psf, seed=seed)
    p = p + n

    i = 10
    p = np.maximum(0.01, p[i])

    shape = (25, 25)
    pl.imshow(p.reshape(shape), interpolation='nearest', origin='lower',
              norm=LogNorm(vmin=p.min(), vmax=p.max()))
    pl.colorbar()
    pl.savefig(plotdir + 'foo.png')
