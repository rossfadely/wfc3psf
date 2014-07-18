import numpy as np

from plotting import plot_data
from generation import render_psfs
from scipy.ndimage.morphology import binary_dilation as grow_mask

def fit_patches(data, dq, shifts, psf_model, parms, old_fit_parms=None):
    """
    Fit the patches and return model components and nll.
    """
    # initialize
    nll = np.zeros_like(data)
    masks = np.zeros_like(data, dtype=np.bool)
    if parms.background == None:            
        fit_parms = np.zeros(data.shape[0])
    elif parms.background == 'constant':
        fit_parms = np.zeros((data.shape[0], 2))
    else:
        assert 0, 'no linear bkgs, need to implement uncertainties'
        fit_parms = np.zeros((data.shape[0], 4))
    fit_vars = np.zeros_like(fit_parms)

    psfs = render_psfs(psf_model, shifts, parms.patch_shape, parms.psf_grid,
                       parms.k)

    for i in range(data.shape[0]):
        dlt_nll = np.inf
        if old_fit_parms == None:
            fp, fv, bkg, ind = fit_single_patch(data[i], psfs[i], dq[i], parms)
        else:
            bkg = make_background(data[i], old_fit_parms[i], parms.background)
            model = old_fit_parms[i][0] * psfs[i] + bkg
            nm = parms.floor + parms.gain * np.abs(model)
            fp, fv, bkg, ind = fit_single_patch(data[i], psfs[i], dq[i], parms,
                                                var=nm)

        model = fp[0] * psfs[i] + bkg
        nm = parms.floor + parms.gain * np.abs(model)
        cur_nll = np.sum(patch_nll(data[i][ind], model[ind], parms))
        cur_fp = fp
        cur_fv = fv
        while dlt_nll > parms.nll_tol:
            fp, fv, bkg, idx = fit_single_patch(data[i], psfs[i], dq[i], parms,
                                           var=nm)
            model = fp[0] * psfs[i] + bkg
            nm = parms.floor + parms.gain * np.abs(model)
            new_nll = np.sum(patch_nll(data[i][idx], model[idx], parms))
            dlt_nll = cur_nll - new_nll
            if dlt_nll > 0:
                ind = idx
                cur_fp = fp
                cur_fv = fv
                cur_nll = new_nll

        assert cur_nll < parms.max_nll
        nll[i, ind] = cur_nll
        masks[i] = ind
        fit_parms[i] = cur_fp
        fit_vars[i] = cur_fv

        if parms.plot_data:
            # get pre-clip nll
            if parms.clip_parms == None:
                parms.old_model = model
                parms.old_bkg = bkg
                flags = dq[i].reshape(parms.patch_shape)
                flags[flags > 1] = 1
                parms.flags = flags
            else:
                ind = parms.flags.ravel() != 1
            old_nll = np.zeros(data.shape[1])
            old_nll[ind] = patch_nll(data[i][ind], parms.old_model[ind], parms)
            
            # plot the data
            t = time.time()
            plot_data(i, data[i], model, bkg, nll[i], old_nll, parms)

    return fit_parms, fit_vars, nll, masks

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

    bkg = make_background(data, fit_parms, background)

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
            bkg = make_background(data, fit_parms, background)

    fit_vars = np.diagonal(lh)

    return fit_parms, fit_vars, bkg, ind

def make_background(data, fit_parms, background):
    """
    Make the backgound model for a patch
    """
    if background is None:
        bkg = 0.0
    elif background == 'constant':
        if len(fit_parms.shape) > 1:
            bkg = np.ones_like(data) * fit_parms[:, -1][:, None]
        else:
            bkg = np.ones_like(data) * fit_parms[-1]
    elif background == 'linear':
        assert 0, 'need to reimplement'
    return bkg
