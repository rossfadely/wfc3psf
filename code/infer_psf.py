import time
import multiprocessing
import numpy as np
import pyfits as pf

from grid_definitions import get_grids
from shifts_update import update_shifts
from patch_fitting import evaluate
from psf_update_linear import update_psf_linear
from psf_update import update_psf
from plotting import psf_plot

def PatchFitter(all_data, all_dq, ini_psf, patch_shape, id_start,
                background='linear',
                sequence=['shifts', 'psf'], tol=1.e-10, eps=1.e-4, gamma=1.e0,
                ini_shifts=None, Nthreads=20, floor=None, plotfilebase=None,
                gain=None, maxiter=np.Inf, dumpfilebase=None, trim_frac=0.005,
                min_data_frac=0.75, core_size=5, lower=1.e-5, k=1,
                plot=False, clip_parms=None, final_clip=[1, 3.], q=1.0,
                clip_shifts=False, h=1.4901161193847656e-08, Nplot=20,
                small=1.e-5, Nsearch=256, search_rate=0.05, search_scale=1e-2,
                shift_test_thresh=0.475, min_frac=0.5, max_nll=1.e10, Nburn=10):
    """
    Patch fitting routines for psf inference.
    """
    assert background in [None, 'constant', 'linear']
    assert np.mod(patch_shape[0], 2) == 1, 'Patch shape[0] must be odd'
    assert np.mod(patch_shape[1], 2) == 1, 'Patch shape[1] must be odd'
    assert np.mod(core_size, 2) == 1, 'Core size must be odd'
    assert (patch_shape[0] * patch_shape[1]) == all_data.shape[1], \
        'Patch shape does not match data shape'
    kinds = ['shifts', 'psf', 'evaluate', 'plot_data']
    for i in range(len(sequence)):
        assert sequence[i] in kinds, 'sequence not allowed'

    # set parameters
    parms = InferenceParms(h, k, q, eps, tol, gain, plot, floor, gamma, all_data.shape[0],
                           Nplot, small, Nsearch, id_start, max_nll, min_frac,
                           Nthreads, core_size, background, None,
                           patch_shape, search_rate, plotfilebase,
                           search_scale, ini_psf.shape, shift_test_thresh,
                           ini_psf)

    # initialize
    current_psf = ini_psf.copy()
    current_psf /= current_psf.max()
    current_cost = np.inf
    if ini_shifts is not None:
        shifts = ini_shifts
        ref_shifts = ini_shifts.copy()
    else:
        ref_shifts = np.zeros((all_data.shape[0], 2))
    if Nburn is not None:
        current_cost = None
        burn_iter = 0
    else:
        data = all_data
        dq = all_dq
        current_cost = np.inf
        burn_iter = None

    # run
    t0 = time.time()
    while True:
        t = time.time()
        # assign data used during burnin
        if burn_iter is not None:
            burn_iter += 1
            if burn_iter == Nburn:
                data = all_data
                dq = all_dq
                current_cost = np.inf
            else:
                burn_size = np.ceil(1. * all_data.shape[0] / Nburn)
                data = all_data[:burn_iter * burn_size]
                dq = all_dq[:burn_iter * burn_size]

        # minimum number of patches, mask initialization
        Nmin = np.ceil(min_data_frac * data.shape[0]).astype(np.int)
        mask = np.arange(data.shape[0], dtype=np.int)
        parms.Ndata = data.shape[0]

        # run a iteration
        for kind in sequence:

            if parms.iter >= maxiter:
                return current_psf

            if kind == 'shifts':
                parms.clip_parms = None
                shifts, nll = update_shifts(data[:, parms.core_ind],
                                            dq[:, parms.core_ind],
                                            current_psf,
                                            np.zeros((data.shape[0], 2)), parms)
                ref_shifts = shifts.copy()

                print 'Shift step 1 done nll, total: ', nll.sum()
                print 'Shift step 1 done nll, min: ', nll.min()
                print 'Shift step 1 done nll, median: ', np.median(nll)
                print 'Shift step 1 done nll, max: ', nll.max()

                if (trim_frac is not None) & (mask.size > Nmin):
                    assert trim_frac > 0., 'trim_frac must be positive or None'
                    Ntrim = np.ceil(mask.size * trim_frac).astype(np.int)
                    if (mask.size - Ntrim < Nmin):
                        Ntrim = mask.size - Nmin

                    # sort and trim the arrays
                    ind = np.sort(np.argsort(nll)[:-Ntrim])
                    dq = dq[ind]
                    data = data[ind]
                    mask = mask[ind]
                    ref_shifts = ref_shifts[ind]
                    parms.Ndata = data.shape[0]
                    parms.data_ids = parms.data_ids[ind]

                    # re-run shifts
                    shifts, nll = update_shifts(data[:, parms.core_ind],
                                                 dq[:, parms.core_ind],
                                                 current_psf, ref_shifts, parms)
                else:
                    ind = np.arange(data.shape[0])

                print 'Shift step 2 done nll, total: ', nll.sum()
                print 'Shift step 2 done nll, min: ', nll.min()
                print 'Shift step 2 done nll, median: ', np.median(nll)
                print 'Shift step 2 done nll, max: ', nll.max()

                if dumpfilebase is not None:
                    name = dumpfilebase + '_mask_%d.dat' % parms.iter
                    np.savetxt(name, mask, fmt='%d')
                    name = dumpfilebase + '_shifts_%d.dat' % parms.iter
                    np.savetxt(name, shifts)
                    name = dumpfilebase + '_shift_nll_%d.dat' % parms.iter
                    np.savetxt(name, nll)

            if kind == 'evaluate':
                parms.return_parms = True
                set_clip_parameters(clip_parms, parms, final_clip)
                nll, fit_parms, masks = evaluate((data, dq, shifts, current_psf,
                                                  parms, False))
                parms.return_parms = False

            if kind == 'psf':
                set_clip_parameters(clip_parms, parms, final_clip)
                if parms.k == 1:
                    new_psf, cost = update_psf_linear(current_psf, data, dq,
                                                      shifts, nll, fit_parms,
                                                      masks, parms)
                else:
                    new_psf, cost = update_psf(current_psf, data, dq, shifts,
                                               nll, fit_parms, masks, parms)

                if new_psf is not None:
                    psf_plot(ini_psf, np.maximum(parms.small, current_psf),
                             np.maximum(parms.small, new_psf), parms.small,
                             parms)
                    current_psf = new_psf
                    if (dumpfilebase is not None):
                        hdu = pf.PrimaryHDU(current_psf / current_psf.max())
                        hdu.writeto(dumpfilebase + '_psf_%d.fits' % parms.iter,
                                    clobber=True)

            if kind == 'plot_data':
                if clip_parms is None:
                    parms.clip_parms = [1, np.inf]
                else:
                    try:
                        parms.clip_parms = clip_parms[parms.iter]
                    except:
                        parms.clip_parms = final_clip

                parms.plot_data = True
                foo = evaluate((data[:parms.Nplot], dq[:parms.Nplot],
                                shifts[:parms.Nplot], current_psf, parms,
                                False))
                parms.plot_data = False

        if current_cost is None:
            tup = (-99., cost)
        else:
            tup = (current_cost, cost)
        print '\n\nUsing %d patches' % data.shape[0]
        print 'Current cost: %0.2e, new cost %0.2e' % tup
        dt = (time.time() - t) / 3600.
        dt0 = (time.time() - t0) / 3600.
        print 'Iter %d took %0.2e hrs, total %0.2e hrs\n\n' % (parms.iter, dt, 
                                                               dt0)
        if current_cost is not None:
            #assert cost < current_cost, 'Global cost did not decrease'
            if np.abs((current_cost - cost) / cost) < tol:
                print 'Converged at cost %s' % cost
                return current_psf
            else:
                current_cost = cost
        parms.iter += 1

class InferenceParms(object):
    """
    Class for storing and referencing parameters used in PSF inference.
    """
    def __init__(self, h, k, q, eps, tol, gain, plot, floor, gamma, Ndata, Nplot, small,
                 Nsearch, id_start, max_nll, min_frac, Nthreads, core_size,
                 background, clip_parms, patch_shape,
                 search_rate, plotfilebase, search_scale, psf_model_shape,
                 shift_test_thresh, ini_psf):
        self.h = h
        self.k = k
        self.q = q
        self.eps = eps
        self.tol = tol
        self.gain = gain
        self.plot = plot
        self.floor = floor
        self.gamma = gamma
        self.Ndata = Ndata
        self.Nplot = Nplot
        self.small = small
        self.ini_psf = ini_psf
        self.max_nll = max_nll
        self.Nsearch = Nsearch
        self.min_frac = min_frac
        self.Nthreads = Nthreads
        self.core_size = core_size
        self.background = background
        self.clip_parms = clip_parms
        self.patch_shape = patch_shape
        self.search_rate = search_rate
        self.plotfilebase = plotfilebase
        self.search_scale = search_scale
        self.psf_model_shape = psf_model_shape
        self.shift_test_thresh = shift_test_thresh

        self.iter = 0
        self.plot_data = False
        self.return_parms = False
        self.data_ids = np.arange(id_start, Ndata + id_start, dtype=np.int)

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
        self.psf_grid, x = get_grids(patch_shape, psf_model_shape)

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
