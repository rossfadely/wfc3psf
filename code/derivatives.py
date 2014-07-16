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
    old_reg, reg_term = reg(psf_model, parms)

    # calculate derivative of nll term
    pool = multiprocessing.Pool(parms.Nthreads)
    mapfn = pool.map

    steps = psf_model.copy() * parms.h
    argslist = [None] * parms.Ndata
    for i in range(parms.Ndata):
        argslist[i] = (data[i], shifts[None, i], psf_model, old_nlls[i],
                       fit_parms[i], masks[i], steps, parms)

    results = list(mapfn(one_datum_nll_diff, [args for args in argslist]))

    Neff = 0
    derivatives = np.zeros_like(psf_model)
    for i in range(parms.Ndata):
        derivatives += results[i]
        if np.any(results[i][0] != 0.0):
            Neff += 1

    if Neff == 0:
        derivatives = np.zeros_like(psf_model)
    else:
        derivatives /= Neff
    derivatives += reg_term

    # tidy up
    pool.close()
    pool.terminate()
    pool.join()

    return derivatives, old_reg

def reg(psf_model, parms):
    """
    Regularization and derivative.
    """
    eps = parms.eps
    if (eps is None):
        return np.zeros_like(psf_model)

    psf_shape = psf_model.shape
    d = np.zeros_like(psf_model)
    r = np.zeros_like(psf_model)
    for i in range(psf_shape[0]):
        for j in range(psf_shape[1]): 
            if i > 0:
                r[i, j] += (psf_model[i, j] - psf_model[i - 1, j]) ** 2.
                d[i, j] += 2. * (psf_model[i, j] - psf_model[i - 1, j]) 
            if j > 0:
                r[i, j] += (psf_model[i, j] - psf_model[i, j - 1]) ** 2.
                d[i, j] += 2. * (psf_model[i, j] - psf_model[i, j - 1]) 
            if i < psf_shape[0] - 1:
                r[i, j] += (psf_model[i, j] - psf_model[i + 1, j]) ** 2.
                d[i, j] += 2. * (psf_model[i, j] - psf_model[i + 1, j]) 
            if j < psf_shape[1] - 1:
                r[i, j] += (psf_model[i, j] - psf_model[i, j + 1]) ** 2.
                d[i, j] += 2. * (psf_model[i, j] - psf_model[i, j + 1]) 
    r *= eps
    d *= eps
    return r, d

def regularization_derivative(psf_model, parms):
    """
    Compute derivative of regularization wrt the psf.
    """
    # old regularization
    old_reg = local_regularization((psf_model, parms, None))

    # Map to the processes
    pool = multiprocessing.Pool(parms.Nthreads)
    mapfn = pool.map

    # compute perturbed reg
    hs = parms.h * psf_model.copy()
    argslist = [None] * parms.psf_model_shape[0] * parms.psf_model_shape[1]
    for i in range(parms.psf_model_shape[0]):
        for j in range(parms.psf_model_shape[1]):
            idx = i * parms.psf_model_shape[1] + j
            tmp_psf = psf_model.copy()
            tmp_psf[i, j] += hs[i, j]
            argslist[idx] = (tmp_psf, parms, (i, j))
    new_reg = np.array((mapfn(local_regularization,
                              [args for args in argslist])))
    new_reg = new_reg.reshape(parms.psf_model_shape)

    # tidy up
    pool.close()
    pool.terminate()
    pool.join()
    
    return (new_reg - old_reg) / hs, old_reg

def one_datum_nll_diff((datum, shift, psf_model, old_nll, fitparms, mask,
                        steps, parms)):
    """
    Calculate the derivative for a single datum using forward differencing.
    """
    # if not enough good pixels, discard patch
    min_pixels = np.ceil(parms.min_frac * datum.size)
    if datum[mask].size < min_pixels:
        return np.zeros_like(psf_model)

    # background model
    if parms.background == 'linear':
        N = np.sqrt(psf_model.size).astype(np.int)
        x, y = np.meshgrid(range(N), range(N))
        A = np.vstack((np.ones_like(psf), np.ones_like(psf),
                       x.ravel(), y.ravel())).T
        bkg = make_background(datum, A, fitparms, parms.background)
    elif parms.background == None:
        bkg = 0.0
    else:
        bkg = fitparms[-1]

    # calculate the difference in nll, tweaking each psf parm.
    steps = parms.h * psf_model
    deriv = np.zeros_like(psf_model)
    for i in range(parms.psf_model_shape[0]):
        for j in range(parms.psf_model_shape[1]):
            temp_psf = psf_model.copy()
            temp_psf[i, j] += steps[i, j]

            psf = render_psfs(temp_psf, shift, parms.patch_shape,
                              parms.psf_grid)[0]

            model = fitparms[0] * psf + bkg
            diff = eval_nll(datum[mask], model[mask], parms) - old_nll[mask]
            deriv[i, j] = np.sum(diff) / steps[i, j]

    return deriv

def local_regularization((psf_model, parms, idx)):
    """
    Calculate the local regularization for each pixel.
    """
    eps = parms.eps
    gamma = parms.gamma
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
    
        # l2 norm
        #reg += gamma * psf_model ** 2.

        # floor
        #reg += 1.e-1 / (1. + np.exp((psf_model - 4e-5) * 2.e5))

    else:
        idx = np.array(idx)
        value = psf_model[idx[0], idx[1]]

        # axis 0
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0 # lower edge case
        ind[ind == psf_shape[0]] = psf_shape[0] - 1 # upper edge case
        diff = psf_model[ind[0], idx[1]] - value
        reg = eps * np.sum(diff ** 2.)

        # axis 1
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0 # lower edge case
        ind[ind == psf_shape[1]] = psf_shape[1] - 1 # upper edge case
        diff = psf_model[idx[0], ind[1]] - value
        reg += eps * np.sum(diff ** 2.)

        # l2 norm
        #reg += gamma * value ** 2.

        # floor
        #reg += 1.e-1 / (1. + np.exp((value - 4e-5) * 2.e5) )

    return reg
