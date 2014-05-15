import multiprocessing
import numpy as np

from patch_fitting import evaluate

def one_data_derivative((datum, dq, shift, psf_model, old_ssqe, old_reg,
                         parms)):
    """
    Calculate the derivative for a single datum using forward differencing.
    """
    counts = np.zeros_like(psf_model)
    derivatives = np.zeros_like(psf_model)
    for i in range(parms.psf_model_shape[0]):
        for j in range(parms.psf_model_shape[1]):
            temp_psf = psf_model.copy()
            temp_psf[i, j] += parms.h

            new_ssqe = evaluate((datum, dq, shift, temp_psf, parms, False))
            new_reg = local_regularization(temp_psf, parms.eps, idx=(i, j))

            derivatives[i, j] = np.sum(new_ssqe - old_ssqe)
            derivatives[i, j] += new_reg - old_reg[i, j]

    ind = np.where(derivatives != 0.0)
    counts[ind] += 1.

    return counts, derivatives

def one_parm_derivative(((i, j), data, dq, shift, psf_model, old_ssqe, old_reg,
                         parms)):
    """
    Calculate the derivative for a single model parameter using forward
    differencing.
    """
    temp_psf = psf_model.copy()
    temp_psf[i, j] += parms.h

    new_ssqe = evaluate((data, dq, shift, temp_psf, parms, False))
    new_reg = local_regularization(temp_psf, parms.eps, idx=(i, j))

    derivative = np.mean(new_ssqe - old_ssqe)
    derivative += new_reg - old_reg

    return derivative / parms.h

def get_derivatives(data, dq, shifts, psf_model, old_costs, old_reg, parms):
    """
    Calculate the derivatives of the objective (in patch_fitting)
    with respect to the psf model.
    """
    assert (len(data.shape) == 2) & (len(dq.shape) == 2), \
        'Data should be the (un)raveled patch'

    # Map to the processes
    pool = multiprocessing.Pool(parms.Nthreads)
    mapfn = pool.map

    # calculate derivative, using specified method across the processes
    if parms.deriv_type == 'data':
        argslist = [None] * parms.Ndata
        for i in range(parms.Ndata):
            argslist[i] = (data[None, i], dq[None, i], shifts[None, i],
                           psf_model, old_costs[i], old_reg, parms)
        results = list(mapfn(one_data_derivative, [args for args in argslist]))

        total_counts = np.zeros_like(psf_model)
        total_derivatives = np.zeros_like(psf_model)
        for i in range(parms.Ndata):
            total_counts += results[i][0]
            total_derivatives += results[i][1]

        derivatives = total_derivatives / parms.Ndata / parms.h

    if parms.deriv_type == 'parameter':
        assert 0, 'possible bug at moment'
        argslist = [None] * parms.psf_model_shape[0] * parms.psf_model_shape[1]
        for i in range(parms.psf_model_shape[0]):
            for j in range(parms.psf_model_shape[1]):
                idx = i * parms.psf_model_shape[1] + j
                argslist[idx] = ((i, j), data, dq, shifts, psf_model, old_costs,
                                 old_reg[i, j], parms)
        results = np.array((mapfn(one_parm_derivative,
                                  [args for args in argslist])))

        derivatives = results.reshape(parms.psf_model_shape)

    # tidy up
    pool.close()
    pool.terminate()
    pool.join()

    return derivatives

def local_regularization(psf_model, eps, idx=None):
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
        ind0 = idx[:, None] + pm[None, :]
        ind0[ind0 == -1] = 0
        ind0[ind0 == psf_shape[0]] = psf_shape[0] - 1
        ind1 = idx[:, None] + pm[None, :]
        ind1[ind1 == -1] = 0
        ind1[ind1 == psf_shape[1]] = psf_shape[1] - 1

        value = psf_model[idx[0], idx[1]]
        reg = eps * np.sum((psf_model[ind0[0], idx[1]] - value) ** 2.)
        reg += eps * np.sum((psf_model[idx[0], ind1[1]] - value) ** 2.)

    return reg
