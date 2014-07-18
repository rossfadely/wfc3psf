import os, sys
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
from matplotlib import cm

sys.path.append(os.environ['FLANN'])
from pyflann import FLANN
from patch_fitting import make_background

def psf_builder(data, masks, shifts, fit_parms, fit_vars, parms):
    """
    Build a psf model by scaling data and using a kernel basis.
    """
    xs = parms.patch_grid[0].ravel()[None, :] + shifts[:, 0, None]
    ys = parms.patch_grid[1].ravel()[None, :] + shifts[:, 1, None]

    bkgs = make_background(data, fit_parms, parms.background)
    scaled = np.abs(data - bkgs) / fit_parms[:, 0][:, None]
    assert parms.background == 'constant', 'need to generalize for diff bkgs'
    scaled_vars = (scaled / fit_parms[:, 0][:, None]) ** 2. * \
        fit_vars[:, 0, None]
    scaled_vars += (1. / fit_parms[:, 0][:, None]) ** 2. * fit_vars[:, -1, None]

    ivars = np.zeros(masks[masks == True].size)
    values = np.zeros(masks[masks == True].size)
    masked_xs = np.zeros(masks[masks == True].size)
    masked_ys = np.zeros(masks[masks == True].size)
    ind = 0
    for i in range(data.shape[0]):
        chunk = masks[i][masks[i] == 1].size
        values[ind: ind + chunk] = scaled[i][masks[i]]
        ivars[ind: ind + chunk] = 1. / scaled_vars[i][masks[i]]
        masked_xs[ind: ind + chunk] = xs[i][masks[i]]
        masked_ys[ind: ind + chunk] = ys[i][masks[i]]
        ind += chunk

    return binned_model(masked_xs, masked_ys, values, ivars, parms.patch_shape,
                        parms.psf_model_shape)
    #return kernel_model(values, masked_xs, masked_ys, parms)

def binned_model(xs, ys, vs, ivars, patch_shape, psf_shape):
    """
    Create model by simple binning.
    """
    assert patch_shape[0] == patch_shape[1], 'assymetric patch not supported'
    assert psf_shape[0] == psf_shape[1], 'assymetric patch not supported'
    patch_side = patch_shape[0]
    Nbin = psf_shape[0]

    v = (patch_side - 1) / 2. + 0.5
    bins = np.linspace(-v, v, Nbin)
    dlt = (bins[1] - bins[0]) / 2.
    binned = np.zeros((Nbin, Nbin))
    for i in range(Nbin):
        for j in range(Nbin):
            ind = (xs <= bins[i] + dlt) & (xs > bins[i] - dlt) & \
                (ys <= bins[j] + dlt) & (ys > bins[j] - dlt)
            if vs[ind].size == 0:
                binned[i, j] = 1.e-5
            else:
                #binned[i, j] = np.median(vs[ind])
                binned[i, j] = np.sum(ivars[ind] * vs[ind]) / np.sum(ivars[ind])
    
    binned /= binned.max()
    return binned

def kernel_model(scaled_data, xs, ys, parms):
    """
    Estimate the value at the target grid given the exemplars, using the 
    specified kernel.
    """
    X = np.vstack((xs, ys,)).T
    x, y = np.meshgrid(parms.psf_grid[0], parms.psf_grid[1])
    T = np.vstack((x.ravel(), y.ravel())).T

    # use flann for distances and indicies
    flann = FLANN()
    p = flann.build_index(X, target_precision=parms.flann_precision,
                          log_level='info')
    inds, dists = flann.nn_index(T, parms.knn, check=p['checks'])

    # go through the grid and compute the model
    model = np.zeros(T.shape[0])
    for i in range(model.size):
        local_values = scaled_data[inds[i]]
        if parms.kernel_parms['type'] == 'gaussian':
            k = np.exp(-1. * dists[i] ** 2. / parms.kernel_parms['gamma'] ** 2.)
            model[i] = np.sum(k * local_values) / np.sum(k)

    return model.reshape(parms.psf_model_shape)
