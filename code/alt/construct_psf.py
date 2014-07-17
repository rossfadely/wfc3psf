import os, sys
import numpy as np

sys.path.append(os.environ['FLANN'])
from pyflann import FLANN
from patch_fitting import make_background

def psf_builder(data, masks, shifts, fit_parms, parms):
    """
    Build a psf model by scaling data and using a kernel basis.
    """
    bkgs = make_background(data, fit_parms, parms.background)
    scaled = (data - bkgs) / fit_parms[:, 0]
    xs = parms.psf_grid[0][None, :] + shifts[:, 0, None, None]
    ys = parms.psf_grid[1][None, :] + shifts[:, 1, None, None]
    values = np.zeros(masks[masks == 1].size)
    masked_xs = np.zeros(masks[masks == 1].size)
    masked_ys = np.zeros(masks[masks == 1].size)
    ind = 0
    for i in range(data.shape[0]):
        chunk = masks[i][masks[i] == 1].size
        values[ind: ind + chunk] = scaled[i][masks[i]]
        masked_xs[ind: ind + chunk] = xs[i][masks[i]]
        masked_ys[ind: ind + chunk] = ys[i][masks[i]]
        ind += chunk

    return kernel_model(values, masked_xs, masked_ys, parms)

def kernel_model(scaled_data, xs, ys, parms):
    """
    Estimate the value at the target grid given the exemplars, using the 
    specified kernel.
    """
    X = np.vstack((xs, ys,)).T
    T = np.vstack((parms.psf_grid[0].ravel(), parms.psf_grid[1].ravel())).T

    # use flann for distances and indicies
    flann = FLANN()
    p = flann.build_index(X, target_precision=parms.flann_precision,
                          log_level='info')
    inds, dists = flann.nn_index(T, parms.knn, check=p['checks'])

    # go throught the grid and compute the model
    model = np.zeros(parms.psf_grid[0].size)
    for i in range(model.size):
        local_values = scaled_data[inds[i]]
        if parms.kernel_parms['type'] == 'gaussian':
            k = np.exp(-1. * dists ** 2. / parms.kernel_parms['gamma'] ** 2.)
            model[i] = np.sum(k * local_values) / np.sum(k)

    return model.reshape(parms.psf_model_shape)
