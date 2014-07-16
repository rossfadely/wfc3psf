import multiprocessing
import numpy as np

from scipy.optimize import fmin_powell
from patch_fitting import patch_nll, fit_single_patch
from generation import render_psfs

def update_shifts(data, dq, psf_model, parms):
    """
    Update the estimate of the subpixel shifts, given current psf and flat.
    """
    # initialize
    p0 = (0., 0.)
    Ndata = data.shape[0]
    nlls = np.zeros(Ndata)
    shifts = np.zeros((Ndata, 2))

    # map to threads
    pool = multiprocessing.Pool(parms.Nthreads)
    mapfn = pool.map
    argslist = [None] * Ndata
    for i in range(Ndata):
        argslist[i] = [p0, psf_model, data[i], dq[i], parms]

    results = list(mapfn(update_single_shift, [args for args in argslist]))
    for i in range(Ndata):
        shifts[i] = results[i][0]
        nlls[i] = results[i][1]

    pool.close()
    pool.terminate()
    pool.join()

    return shifts, nlls

def update_single_shift((p0, psf_model, datum, dq, parms)):
    """
    Update a single shift
    """
    # fmin or fmin_powell seems to perform better than
    # fmin_bfgs or fmin_l_bfgs_b.  Powell seems to be as
    # good as fmin, and quicker.
    res = fmin_powell(shift_loss, p0, full_output=True, disp=False,
               args=(psf_model, datum, dq, parms))
    print res
    # if shift is near an edge test to see if it likes other side
    shift = res[0].copy()
    ind = np.abs(shift) > parms.shift_test_thresh
    if np.any(ind):
        new_p0 = shift
        new_p0[ind] *= -1.
        new_res = fmin_powell(shift_loss, new_p0, full_output=True, disp=False,
                              args=(psf_model, datum, dq, parms))

        if (new_res[1] < res[1]):
            return new_res
    return res

def shift_loss(shift, psf_model, datum, dq, parms):
    """
    Evaluate the shift for a given patch.
    """
    # Horrible hack for minimizers w/o bounds
    if np.any(np.abs(shift) > 0.5):
        return parms.max_nll

    # render the psf model for given shift
    psf = render_psfs(psf_model, shift[None, :], parms.core_shape,
                      parms.psf_grid, parms.k)

    # the fit
    fit_parms, bkg, ind = fit_single_patch(datum, psf[0], dq, parms)
    model = fit_parms[0] * psf[0] + bkg

    return np.sum(patch_nll(datum[ind], model[ind], parms))
