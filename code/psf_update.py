import multiprocessing
import numpy as np

from grid_definitions import get_grids
from patch_fitting import eval_nll, evaluate, fit_single_patch
from derivatives import get_derivatives, local_regularization
from generation import render_psfs
from plotting import searchplot

def update_psf(psf_model, data, dq, shifts, old_nll, fit_parms, masks,
               parms):
    """
    Update the psf model by calculating numerical derivatives and 
    finding the appropriate step in those directions.
    """
    # heavy lifting, get derivatives
    derivatives, old_reg = get_derivatives(data, dq, shifts, psf_model, old_nll,
                                           fit_parms, masks, parms)

    # check that small step improves the model
    temp_psf = psf_model.copy() - derivatives * parms.h
    nll = evaluate((data, dq, shifts, temp_psf, parms, False))
    nll = np.mean(nll[nll < parms.max_nll])
    reg = np.sum(local_regularization((temp_psf, parms.eps, None)))
    old_nll = np.mean(old_nll[old_nll < parms.max_nll])
    old_reg = np.sum(old_reg)
    old_cost = old_nll + old_reg
    assert (nll + reg - old_cost) < 0.0, 'psf update error'

    # find update to the psf
    current_scale = parms.search_scale
    regs = np.zeros(parms.Nsearch)
    nlls = np.zeros(parms.Nsearch)
    costs = np.zeros(parms.Nsearch)
    scales = np.zeros(parms.Nsearch)
    best_cost = np.inf
    for i in range(parms.Nsearch):
        # perturb
        temp_psf = psf_model.copy() - derivatives * current_scale
        temp_psf = np.maximum(parms.small, temp_psf)

        # evaluate
        nll = evaluate((data, dq, shifts, temp_psf, parms, False))
        nll = np.mean(nll[nll < parms.max_nll])
        reg = np.sum(local_regularization((temp_psf, parms.eps, None)))

        # store
        regs[i] = reg
        nlls[i] = nll
        costs[i] = reg + nll
        scales[i] = current_scale

        # update best
        if costs[i] < best_cost:
            msg = 'Search step %d: nll: %0.4e, reg: %0.4e, cost: %0.4e, ' + \
                'scale: %0.4e'
            print msg % (i, nlls[i], regs[i], costs[i], current_scale)
            best_reg = regs[i]
            best_nll = nlls[i]
            best_cost = costs[i]
            best_scale = scales[i]

        # go down in scale
        current_scale = np.exp(np.log(current_scale) - parms.search_rate)

    # fill in broken searches
    ind = nlls == np.inf
    idx = nlls != np.inf
    nlls[ind] = np.max(nlls[idx])

    print 'Old nll: %0.4e, reg: %0.4e, total: %0.4e' % \
        (old_nll, old_reg, old_cost)
    print 'New nll: %0.4e, reg: %0.4e, total: %0.4e, at scale %0.3e' % \
        (best_nll, best_reg, best_cost, best_scale)
    
    # update
    psf_model = psf_model - derivatives * best_scale
    psf_model = np.maximum(parms.small, psf_model)

    if parms.plot:
        searchplot(nlls, regs, scales, parms)

    return psf_model, best_cost
