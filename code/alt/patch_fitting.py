import numpy as np

from plotting import plot_data
from generation import render_psfs
from scipy.ndimage.morphology import binary_dilation as grow_mask

def patch_nll(data, model, parms):
    """
    Return the specified error/loss/nll.
    """
    var = parms.floor + parms.gain * np.abs(model)
    nll = 0.5 * (np.log(var) + (data - model) ** 2. / var)
    return nll

def fit_single_patch(data, psf, dq, parms, var=None):
    """
    Fit a single patch, return the scale for the psf plus any
    background parameters.  Takes in flattened arrays for
    data and psf.
    """
    gain = parms.gain
    floor = parms.floor
    clip_parms = parms.clip_parms
    background = parms.background

    if var == None:
        var = np.ones_like(data)

    if background is None:
        A = np.atleast_2d(psf).T
    elif background == 'constant':
        A = np.vstack((psf, np.ones_like(psf))).T
    elif background == 'linear':
        N = np.sqrt(psf.size).astype(np.int)
        x, y = np.meshgrid(range(N), range(N))
        A = np.vstack((psf, np.ones_like(psf),
                       x.ravel(), y.ravel())).T
    else:
        assert False, 'Background model not supported: %s' % background

    ind = dq == 0

    # fit the data using least squares
    rh = np.dot(A[ind, :].T, data[ind] / var[ind])
    try:
        lh = np.linalg.inv(np.dot(A[ind, :].T, A[ind, :] / var[ind, None]))
        fit_parms = np.dot(lh, rh)
    except:
        fit_parms = np.zeros(A.shape[1])

    bkg = make_background(data, A, fit_parms, background)

    # sigma clip if desired
    if (clip_parms is not None) & (np.any(fit_parms != 0)):
        Niter = clip_parms[0]
        assert Niter == 1, 'Multiple rounds of clipping not supported.'
        tol = clip_parms[1]
        for i in range(Niter):

            # define model and noise
            scaled_psf = psf * fit_parms[0]
            model = scaled_psf + bkg
            if parms.plot_data:
                parms.old_bkg = bkg
                parms.old_model = model
            model = model[ind]
            scaled_psf = scaled_psf[ind]
            var = floor + gain * np.abs(model) + (parms.q * scaled_psf) ** 2.

            # sigma clip
            chi = np.zeros_like(data)
            chi[ind] = np.abs(data[ind] - model) / np.sqrt(var)
            condition = chi - tol
            condition = (condition > 0).reshape(parms.patch_shape)
            
            # redefine mask, grow and add to dq mask.
            ind = 1 - ind.reshape(parms.patch_shape)
            idx = grow_mask(condition)
            if parms.plot_data:
                parms.flags = ind.copy()
                parms.flags[idx] = 3
                parms.flags[condition] = 2
            ind = np.ravel((ind == 0) & (idx == 0))

            # refit
            rh = np.dot(A[ind, :].T, data[ind])
            try:
                lh = np.linalg.inv(np.dot(A[ind, :].T, A[ind, :]))
                fit_parms = np.dot(lh, rh)
            except:
                fit_parms = np.zeros(A.shape[1])
            bkg = make_background(data, A, fit_parms, background)

    return fit_parms, bkg, ind

def make_background(data, A, fit_parms, background):
    """
    Make the backgound model for a patch
    """
    bkg = np.zeros_like(data)
    if background is not None:
        for i in range(A.shape[1] - 1):
            bkg += A[:, i + 1] * fit_parms[i + 1]
    return bkg
