import numpy as np

from construct_psf import psf_builder
from patch_fitting import fit_patches
from shifts_update import update_shifts
from grid_definitions import get_grids

import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
from matplotlib import cm

def learn_psf(data, dq, initial_psf, clip_parms, noise_parms, 
              plotfilebase, kernel_parms, patch_shape, knn=32,
              min_patch_frac=0.75, core_size=5, nll_tol=1.e-5,
              k=3, q=1.0, Nplot=20, plot=False, flann_precision=0.99,
              final_clip=[1, 3.], background='constant', Nthreads=20,
              max_iter=20, max_nll=1.e10, shift_test_thresh=0.475):
    """
    Inference routine for learning a psf model via scaled data and a 
    kernel basis.
    """
    assert background in [None, 'constant', 'linear']
    assert np.mod(patch_shape[0], 2) == 1, 'Patch shape[0] must be odd'
    assert np.mod(patch_shape[1], 2) == 1, 'Patch shape[1] must be odd'
    assert np.mod(core_size, 2) == 1, 'Core size must be odd'
    assert (patch_shape[0] * patch_shape[1]) == data.shape[1], \
        'Patch shape does not match data shape'

    # bundle parameters to be passed to other functions
    parms = InferenceParms(k, q, knn, plot, data.shape[0], Nplot, nll_tol,
                           max_nll, Nthreads,
                           core_size, background, clip_parms, patch_shape,
                           noise_parms, kernel_parms, plotfilebase,
                           min_patch_frac, flann_precision,
                           initial_psf.shape, shift_test_thresh)

    # initialize
    print 'Initialized with %d patches\n' % data.shape[0]
    initial_psf /= initial_psf.max()
    psf_model = initial_psf.copy()
    cost = np.Inf

    # Run through data, reject patches that are bad/crowded.
    parms.clip_parms = None
    shifts, nll = update_shifts(data[:, parms.core_ind], dq[:, parms.core_ind],
                                psf_model, parms)
    set_clip_parameters(clip_parms, parms, final_clip)
    fit_parms, fit_vars, nll, masks = fit_patches(data, dq, shifts, psf_model,
                                                  parms)
    nll = np.sum(nll, axis=1)
    ind = nll < parms.max_nll
    shifts = shifts[ind]
    data = data[ind]
    dq = dq[ind]
    parms.data_ids = data[ind]
    print '%d patches are ok under the initial model\n' % data.shape[0]
    print 'Initial NLL is %0.6e' % np.sum(nll)

    # Build a new psf
    psf_model = psf_builder(data, masks, shifts, fit_parms, fit_vars, parms)

    for blah in range(12):
        parms.clip_parms = None
        shifts, nll = update_shifts(data[:, parms.core_ind],
                                    dq[:, parms.core_ind],
                                    psf_model, parms)
        print blah, nll.sum(), shifts[0], shifts[-1]
        set_clip_parameters(clip_parms, parms, final_clip)
        fit_parms, fit_vars, nll, masks = fit_patches(data, dq, shifts,
                                                      psf_model,
                                                      parms)
        nll = np.sum(nll, axis=1)
        ind = nll < parms.max_nll
        shifts = shifts[ind]
        data = data[ind]
        dq = dq[ind]
        parms.data_ids = data[ind]
        print 'New NLL is %0.6e' % np.sum(nll)
        f=pl.figure(figsize=(16, 8))
        pl.subplot(121)
        pl.imshow(np.abs(psf_model), interpolation='nearest', origin='lower',
                  norm=LogNorm(vmin=1.e-6, vmax=1.))
        psf_model = psf_builder(data, masks, shifts, fit_parms, fit_vars, parms)
        pl.colorbar(shrink=0.7)
        pl.subplot(122)
        pl.imshow(np.abs(psf_model), interpolation='nearest', origin='lower',
                  norm=LogNorm(vmin=1.e-6, vmax=1.))
        pl.colorbar(shrink=0.7)
        print psf_model[50, 50]
        print psf_model.max()
        f.savefig('../../plots/foo.png')
    assert 0

class InferenceParms(object):
    """
    Class for storing and referencing parameters used in PSF inference.
    """
    def __init__(self, k, q, knn, plot, Ndata, Nplot, nll_tol, max_nll, Nthreads,
                 core_size, background, clip_parms, patch_shape,
                 noise_parms, kernel_parms, plotfilebase, min_patch_frac,
                 flann_precision, psf_model_shape, 
                 shift_test_thresh):
        self.k = k
        self.q = q
        self.knn = knn
        self.gain = noise_parms['gain']
        self.plot = plot
        self.floor = noise_parms['floor']
        self.Nplot = Nplot
        self.nll_tol = nll_tol
        self.max_nll = max_nll
        self.Nthreads = Nthreads
        self.core_size = core_size
        self.background = background
        self.clip_parms = clip_parms
        self.patch_shape = patch_shape
        self.kernel_parms = kernel_parms
        self.plotfilebase = plotfilebase
        self.flann_precision = flann_precision
        self.psf_model_shape = psf_model_shape
        self.shift_test_thresh = shift_test_thresh

        self.iter = 0
        self.plot_data = plot
        self.return_parms = False
        self.data_ids = np.arange(0, Ndata, dtype=np.int)
        self.min_pixels = np.ceil(min_patch_frac * patch_shape[0] *
                                  patch_shape[1])

        self.set_grids(core_size, patch_shape, psf_model_shape)

        xsamp = (psf_model_shape[0] - 1) / patch_shape[0]
        ysamp = (psf_model_shape[1] - 1) / patch_shape[1]
        self.subsample = np.array([xsamp, ysamp])
        xstep = self.psf_grid[0][1] - self.psf_grid[0][0]
        ystep = self.psf_grid[1][1] - self.psf_grid[1][0]
        self.psf_steps = np.array([xstep, ystep])

    def set_grids(self, core_size, patch_shape, psf_model_shape):
        """
        Set grid definitions for PSF and patches
        """
        # core foo
        ravel_size = patch_shape[0] * patch_shape[1]
        self.core_shape = (core_size, core_size)
        xcenter = (patch_shape[0] - 1) / 2
        ycenter = (patch_shape[1] - 1) / 2
        buff = (core_size - 1) / 2
        xcore = xcenter - buff, xcenter + buff + 1
        ycore = ycenter - buff, ycenter + buff + 1
        core_ind = np.arange(ravel_size, dtype=np.int).reshape(patch_shape)
        self.core_ind = core_ind[xcore[0]:xcore[1], ycore[0]:ycore[1]].ravel()

        # grid defs
        self.psf_grid, self.patch_grid = get_grids(patch_shape, psf_model_shape)

def set_clip_parameters(clip_parms, parms, final_clip):
    """
    Set clipping, used during full patch fitting.
    """
    if clip_parms is None:
        parms.clip_parms = None
    else:
        try:
            parms.clip_parms = clip_parms[parms.iter]
        except:
            parms.clip_parms = final_clip
