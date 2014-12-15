import numpy as np
from scipy.interpolate import RectBivariateSpline

class CoreModel(object):
    """
    A model for the core of the psf, defined on a rectangular grid.  This is
    used only to infer the subpixel shifts of psf model, since it is much
    faster to do cubic spline interp than render a new GP at each shift.
    """
    def __init__(self, initial_psf, parms):
        assert initial_psf.shape[0] == initial_psf.shape[1], 'square psf only'
        self.core_shape = (parms.core_size, parms.core_size)


        # define indices of core for a flattened patch
        self.core_ind = self.get_core_inds(parms.patch_shape,
                                           parms.core_size).ravel()

        # define indices of core for the psf
        size = 1 + parms.core_size * (initial_psf.shape[0] - 1) / \
               parms.patch_shape[0]
        self.psf_core_ind = self.get_core_inds(initial_psf.shape, size)

        # grid defs
        self.core_rng, self.core_xy = self.xy_grid(parms.core_size)
        r, xy = self.xy_grid(1 + parms.core_size, self.psf_core_ind.shape[0])
        self.core_psf_rng, self.core_psf_xy = r, xy

        # set the initial model
        self.initial_core_model = initial_psf.ravel()[self.psf_core_ind]
        self.core_model = self.initial_core_model.copy()

    def get_core_inds(self, big_shape, core_size):
        """
        Construct the indicies for the core, either for a patch or the psf 
        model.
        """
        center = (big_shape[0] - 1) / 2
        buff = (core_size - 1) / 2
        core = center - buff, center + buff + 1
        core_ind = np.arange(big_shape[0] * big_shape[1],
                             dtype=np.int).reshape(big_shape)
        return core_ind[core[0]:core[1], core[0]:core[1]]

    def xy_grid(self, size, N=None):
        """
        Return 1D range and 2D grid over given size. 
        """
        if N is None:
            N = size

        g = np.linspace(-0.5 * (size - 1), 0.5 * (size - 1), N)
        xg, yg = np.meshgrid(g, g)
        return g, np.vstack((xg.flatten(), yg.flatten())).T

    def render_core_psfs(self, shifts, parms):
        """
        Render the psf model.
        """
        interp_func = RectBivariateSpline(self.core_psf_rng, self.core_psf_rng,
                                          self.core_model, kx=parms.k,
                                          ky=parms.k)
        psfs = np.zeros((shifts.shape[0], parms.core_size ** 2))
        for i in range(psfs.shape[0]):
            psfs[i] = interp_func(self.core_rng + shifts[i, 0],
                                  self.core_rng + shifts[i, 1]).T.ravel()
        return psfs

if __name__ == '__main__':

    import pyfits as pf
    from utils import FitParms
    from matplotlib import use; use('Agg')
    import matplotlib.pyplot as pl

    parms = FitParms()
    f = pf.open('../../psfs/tinytim-pixelconvolved-507-507-25-25-101-101.fits')
    ini = f[0].data
    f.close()

    cm = CoreModel(ini, parms)

    f = pl.figure(figsize=(10, 5))
    pl.gray()
    pl.subplot(121)
    pl.imshow(np.log(cm.core_model), interpolation='nearest', origin='lower')
    pl.subplot(122)
    pl.imshow(np.log(ini), interpolation='nearest', origin='lower',
              vmin=np.log(cm.core_model).min())
    f.savefig('../../plots/foo.png')
    pl.close()

    psfs = cm.render_core_psfs(np.array([[-0.25, 0.25], [0., 0.]]), parms)

    f = pl.figure(figsize=(10, 5))
    pl.gray()
    pl.subplot(121)
    pl.imshow(np.log(psfs[0].reshape(5, 5)), interpolation='nearest',
              origin='lower')
    pl.subplot(122)
    pl.imshow(np.log(psfs[1].reshape(5, 5)), interpolation='nearest',
              origin='lower')
    f.savefig('../../plots/foo.png')
    pl.close()
