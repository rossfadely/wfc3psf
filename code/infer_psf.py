import multiprocessing
import numpy as np
import pyfits as pf

from grid_definitions import get_grids
from shifts_update import update_shifts
from patch_fitting import evaluate
from psf_update import update_psf

def PatchFitter(data, dq, ini_psf, patch_shape, id_start, background='linear',
                sequence=['shifts', 'psf'], tol=1.e-4, eps=1.e-4,
                ini_shifts=None, Nthreads=20, floor=None, plotfilebase=None,
                gain=None, maxiter=np.Inf, dumpfilebase=None, trim_frac=0.005,
                min_data_frac=0.75, core_size=5,
                plot=False, clip_parms=None, final_clip=[1, 3.], q=1.0,
                clip_shifts=False, h=1.4901161193847656e-08, Nplot=20,
                small=1.e-6, Nsearch=64, search_rate=0.125, search_scale=1e-4,
                shift_test_thresh=0.475, min_frac=0.5, max_ssqe=1.e10):
    """
    Patch fitting routines for BallPeenHammer.
    """
    assert background in [None, 'constant', 'linear']
    assert np.mod(patch_shape[0], 2) == 1, 'Patch shape[0] must be odd'
    assert np.mod(patch_shape[1], 2) == 1, 'Patch shape[1] must be odd'
    assert np.mod(core_size, 2) == 1, 'Core size must be odd'
    assert (patch_shape[0] * patch_shape[1]) == data.shape[1], \
        'Patch shape does not match data shape'
    kinds = ['shifts', 'psf', 'evaluate', 'plot_data']
    for i in range(len(sequence)):
        assert sequence[i] in kinds, 'sequence not allowed'

    # set parameters
    parms = InferenceParms(h, q, eps, tol, gain, plot, floor, data.shape[0],
                           Nplot, small, Nsearch, id_start, max_ssqe, min_frac,
                           Nthreads, core_size, background, None,
                           patch_shape, search_rate, plotfilebase,
                           search_scale, ini_psf.shape, shift_test_thresh)

    # initialize
    current_psf = ini_psf.copy()
    current_psf /= current_psf.max()
    current_loss = np.Inf
    if ini_shifts is not None:
        shifts = ini_shifts
        ref_shifts = ini_shifts.copy()
    else:
        ref_shifts = np.zeros((data.shape[0], 2))

    # minimum number of patches, mask initialization
    Nmin = np.ceil(min_data_frac * data.shape[0]).astype(np.int)
    mask = np.arange(data.shape[0], dtype=np.int)

    # run
    cost = np.Inf
    tot_cost = np.Inf
    while True:
        for kind in sequence:

            if parms.iter >= maxiter:
                return current_psf

            if kind == 'shifts':
                parms.clip_parms = None
                shifts, ssqe = update_shifts(data[:, parms.core_ind],
                                             dq[:, parms.core_ind],
                                             current_psf, ref_shifts, parms)

                if parms.iter == 0:
                    ref_shifts = shifts.copy()

                print 'Shift step 1 done ssqe, total: ', ssqe.sum()
                print 'Shift step 1 done ssqe, min: ', ssqe.min()
                print 'Shift step 1 done ssqe, median: ', np.median(ssqe)
                print 'Shift step 1 done ssqe, max: ', ssqe.max()

                if (trim_frac is not None) & (mask.size > Nmin):
                    assert trim_frac > 0., 'trim_frac must be positive or None'
                    Ntrim = np.ceil(mask.size * trim_frac).astype(np.int)
                    if (mask.size - Ntrim < Nmin):
                        Ntrim = mask.size - Nmin

                    # sort and trim the arrays
                    ind = np.sort(np.argsort(ssqe)[:-Ntrim])
                    dq = dq[ind]
                    data = data[ind]
                    mask = mask[ind]
                    ref_shifts = ref_shifts[ind]
                    parms.Ndata = data.shape[0]
                    parms.data_ids = parms.data_ids[ind]

                    # re-run shifts
                    shifts, ssqe = update_shifts(data[:, parms.core_ind],
                                                 dq[:, parms.core_ind],
                                                 current_psf, ref_shifts, parms)
                else:
                    ind = np.arange(data.shape[0])

                print 'Shift step 2 done ssqe, total: ', ssqe.sum()
                print 'Shift step 2 done ssqe, min: ', ssqe.min()
                print 'Shift step 2 done ssqe, median: ', np.median(ssqe)
                print 'Shift step 2 done ssqe, max: ', ssqe.max()

                if dumpfilebase is not None:
                    name = dumpfilebase + '_mask_%d.dat' % parms.iter
                    np.savetxt(name, mask, fmt='%d')
                    name = dumpfilebase + '_shifts_%d.dat' % parms.iter
                    np.savetxt(name, shifts)
                    name = dumpfilebase + '_shift_ssqe_%d.dat' % parms.iter
                    np.savetxt(name, ssqe)

            if kind == 'evaluate':
                parms.return_parms = True
                nll, parms = evaluate((data, dq, shifts, current_psf, parms,
                                       False))
                parms.retrun_parms = False

            if kind == 'psf':
                set_clip_parameters(clip_parms, parms, final_clip)
                new_psf = update_psf(current_psf, data, dq, shifts,
                                     validation_data, validation_dq, v_shifts,
                                     parms)

                if new_psf is not None:
                    current_psf = new_psf
                    if (dumpfilebase is not None):
                        hdu = pf.PrimaryHDU(current_psf)
                        hdu.writeto(dumpfilebase + '_psf_%d.fits' % parms.iter,
                                    clobber=True)

        parms.iter += 1

class InferenceParms(object):
    """
    Class for storing and referencing parameters used in PSF inference.
    """
    def __init__(self, h, q, eps, tol, gain, plot, floor, Ndata, Nplot, small,
                 Nsearch, id_start, max_ssqe, min_frac, Nthreads, core_size,
                 background, clip_parms, patch_shape,
                 search_rate, plotfilebase, search_scale, psf_model_shape,
                 shift_test_thresh):
        self.h = h
        self.q = q
        self.eps = eps
        self.tol = tol
        self.gain = gain
        self.plot = plot
        self.floor = floor
        self.Ndata = Ndata
        self.Nplot = Nplot
        self.small = small
        self.Nsearch = Nsearch
        self.max_ssqe = max_ssqe
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