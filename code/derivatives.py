import multiprocessing
import numpy as np

from patch_fitting import eval_nll, make_background, evaluate
from generation import render_psfs

def get_derivatives(data, dq, shifts, psf_model, old_nlls, fit_parms, masks,
                    parms):
    """
    Calculate the derivatives of the objective (in patch_fitting)
    with respect to the psf model.
    """
    # derivative of regularization term
    reg_term, old_reg = regularization_derivative(psf_model, parms)

    # calculate derivative of nll term
    pool = multiprocessing.Pool(parms.Nthreads)
    mapfn = pool.map

    argslist = [None] * parms.Ndata
    for i in range(parms.Ndata):
        argslist[i] = (data[i], shifts[None, i], psf_model, old_nlls[i],
                       fit_parms[i], masks[i], parms)

    results = list(mapfn(one_datum_nll_diff, [args for args in argslist]))

    Neff = 0
    nll_diff_sums = np.zeros_like(psf_model)
    for i in range(parms.Ndata):
        nll_diff_sums += results[i]
        if np.any(results[i][0] != 0.0):
            Neff += 1

    if Neff == 0:
        derivatives = np.zeros_like(psf_model)
    else:
        derivatives = nll_diff_sums / Neff / parms.h + reg_term

    # tidy up
    pool.close()
    pool.terminate()
    pool.join()

    return derivatives, old_reg

def regularization_derivative(psf_model, parms):
    """
    Compute derivative of regularization wrt the psf.
    """
    # old regularization
    old_reg = local_regularization((psf_model, parms.eps, None))

    # Map to the processes
    pool = multiprocessing.Pool(parms.Nthreads)
    mapfn = pool.map

    # compute perturbed reg
    argslist = [None] * parms.psf_model_shape[0] * parms.psf_model_shape[1]
    for i in range(parms.psf_model_shape[0]):
        for j in range(parms.psf_model_shape[1]):
            idx = i * parms.psf_model_shape[1] + j
            tmp_psf = psf_model.copy()
            tmp_psf[i, j] += parms.h
            argslist[idx] = (tmp_psf, parms.eps, (i, j))
    new_reg = np.array((mapfn(local_regularization,
                              [args for args in argslist])))

    # tidy up
    pool.close()
    pool.terminate()
    pool.join()

    return (new_reg.reshape(parms.psf_model_shape) - old_reg) / parms.h, old_reg

def one_datum_nll_diff((datum, shift, psf_model, old_nll, fitparms, mask,
                        parms)):
    """
    Calculate the derivative for a single datum using forward differencing.
    """
    # if not enough good pixels, discard patch
    min_pixels = np.ceil(parms.min_frac * datum.size)
    if datum[mask].size < min_pixels:
        return np.zeros_like(psf_model)

    # background model
    if parms.background == 'linear':
        N = np.sqrt(psf.size).astype(np.int)
        x, y = np.meshgrid(range(N), range(N))
        A = np.vstack((psf, np.ones_like(psf),
                       x.ravel(), y.ravel())).T
        bkg = make_background(datum, A, fitparms, parms.background)
    else:
        bkg = fitparms[-1]

    # calculate the difference in nll, tweaking each psf parm.
    nll_diff = np.zeros_like(psf_model)
    for i in range(parms.psf_model_shape[0]):
        for j in range(parms.psf_model_shape[1]):
            temp_psf = psf_model.copy()
            temp_psf[i, j] += parms.h

            psf = render_psfs(temp_psf, shift, parms.patch_shape,
                              parms.psf_grid)[0]

            model = fitparms[0] * psf + bkg
            new_nll = eval_nll(datum[mask], model[mask], parms)
            nll_diff[i, j] = np.sum(new_nll - old_nll[mask])

    return nll_diff

def local_regularization((psf_model, eps, idx)):
    """
    Calculate the local regularization for each pixel.
    """
    if (eps is None):
        if idx is None:
            return np.zeros_like(psf_model)
        else:
            return 0.0

    pm = np.array([-1, 1])
    psf_shape = psf_model.shape
    reg = np.zeros_like(psf_model)

    if idx is None:
        # axis 0
        idx = np.arange(psf_shape[0])
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0 # boundary foo
        ind[ind == psf_shape[0]] = psf_shape[0] - 1 # boundary foo
        for i in range(psf_shape[1]):
            diff = psf_model[ind, i] - psf_model[idx, i][:, None]
            reg[:, i] += eps * np.sum(diff ** 2., axis=1)

        # axis 1
        idx = np.arange(psf_shape[1])
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0 # boundary foo
        ind[ind == psf_shape[1]] = psf_shape[1] - 1 # boundary foo
        for i in range(psf_shape[0]):
            diff = psf_model[i, ind] - psf_model[i, idx][:, None]
            reg[i, :] += eps * np.sum(diff ** 2., axis=1)

    else:
        idx = np.array(idx)
        value = psf_model[idx[0], idx[1]]

        # axis 0
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0 # lower edge case
        ind[ind == psf_shape[0]] = psf_shape[0] - 1 # upper edge case
        reg = eps * np.sum((psf_model[ind[0], idx[1]] - value) ** 2.)

        # axis 1
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0 # lower edge case
        ind[ind == psf_shape[1]] = psf_shape[1] - 1 # upper edge case
        reg += eps * np.sum((psf_model[idx[0], ind[1]] - value) ** 2.)

    return reg
