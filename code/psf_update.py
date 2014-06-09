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
    reg = local_regularization((temp_psf, parms, None))
    reg = np.sum(reg)
    old_nll = np.mean(old_nll[old_nll < parms.max_nll])
    old_reg = np.sum(old_reg)
    old_cost = old_nll + old_reg
    print old_nll, old_reg, old_cost
    print nll, reg, nll + reg
    assert (nll + reg - old_cost) < 0.0, 'psf update error'

    # find update to the psf
    Nbad = 0
    nlls = []
    regs = []
    costs = []
    scales = []
    search = False
    current_scale = parms.search_scale
    best_cost = np.inf
    while True:
        # perturb
        temp_psf = psf_model.copy() - derivatives * current_scale
        #temp_psf = np.maximum(parms.small, temp_psf)

        # evaluate
        nll = evaluate((data, dq, shifts, temp_psf, parms, False))
        nll = np.mean(nll[nll < parms.max_nll])
        reg = local_regularization((temp_psf, parms, None))
        reg = np.sum(reg)
        cost = reg + nll

        #if np.any(temp_psf < 0.0):
        #    cost = parms.max_nll

        # store
        regs.append(reg)
        nlls.append(nll)
        costs.append(cost)
        scales.append(current_scale)

        # update best
        if cost < best_cost:
            msg = 'Search step: nll: %0.4e, reg: %0.4e, cost: %0.4e, ' + \
                'scale: %0.4e'
            print msg % (nll, reg, cost, current_scale)
            best_reg = reg
            best_nll = nll
            best_cost = cost
            best_scale = current_scale
            search = True
            if current_scale < parms.h:
                break
        elif search:
            Nbad += 1
            if Nbad == parms.Nsearch:
                break

        # go down in scale
        current_scale = np.exp(np.log(current_scale) - parms.search_rate)

    print 'Old nll: %0.4e, reg: %0.4e, total: %0.4e' % \
        (old_nll, old_reg, old_cost)
    print 'New nll: %0.4e, reg: %0.4e, total: %0.4e, at scale %0.3e' % \
        (best_nll, best_reg, best_cost, best_scale)
    assert best_cost < old_cost, 'Update search failed.'

    # update
    psf_model = psf_model - derivatives * best_scale
    psf_model = np.maximum(parms.small, psf_model)
    psf_model /= psf_model.max()

    if parms.plot:
        searchplot(np.array(nlls), np.array(regs), np.array(scales), parms)

    return psf_model, best_cost
