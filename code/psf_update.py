import multiprocessing
import numpy as np

from grid_definitions import get_grids
from patch_fitting import eval_nll, evaluate, fit_single_patch
from derivatives import get_derivatives, local_regularization
from generation import render_psfs
from plotting import searchplot

def update_psf(psf_model, data, dq,  shifts, valid_data, valid_dq, valid_shifts,
               parms):
    """
    Update the psf model by calculating numerical derivatives and 
    finding the appropriate step in those directions.
    """
    # get current regularization term
    Ndata = data.shape[0]
    old_reg = local_regularization(psf_model, parms.eps)

    # get current scaled squared error
    old_ssqe = evaluate((data, dq, shifts, psf_model, parms, False))

    # heavy lifting, get derivatives
    derivatives = get_derivatives(data, dq, shifts, psf_model, old_ssqe,
                                  old_reg, parms)
    old_reg = old_reg.sum()

    # get validation ssqe
    old_valid_ssqe = evaluate((valid_data, valid_dq, valid_shifts, psf_model,
                               parms, False))
    old_valid_ssqe = validation_score(old_valid_ssqe, parms)
    old_cost = old_reg + old_valid_ssqe

    # check that small step improves the model
    temp_psf = psf_model.copy() - derivatives * parms.h
    valid_ssqe = evaluate((valid_data, valid_dq, valid_shifts, temp_psf,
                           parms, False))
    valid_ssqe = validation_score(valid_ssqe, parms)
    reg = local_regularization(temp_psf, parms.eps)
    if (valid_ssqe + np.sum(reg) - old_cost) > 0.0:
        print '\n\n\nNo PSF update\n\n\n'
        return None

    # find update to the psf
    current_scale = parms.search_scale
    regs = np.zeros(parms.Nsearch)
    ssqes = np.zeros(parms.Nsearch)
    costs = np.zeros(parms.Nsearch)
    scales = np.zeros(parms.Nsearch)
    best_cost = np.inf
    for i in range(parms.Nsearch):
        # perturb
        temp_psf = psf_model.copy() - derivatives * current_scale
        temp_psf = np.maximum(parms.small, temp_psf)

        # evaluate
        valid_ssqe = evaluate((valid_data, valid_dq, valid_shifts, temp_psf,
                               parms, False))
        valid_ssqe = validation_score(valid_ssqe, parms)
        reg = np.sum(local_regularization(temp_psf, parms.eps))
        # store
        regs[i] = reg
        ssqes[i] = valid_ssqe
        costs[i] = reg + valid_ssqe
        scales[i] = current_scale

        # update best
        if costs[i] < best_cost:
            msg = 'Search step %d: ssqe: %0.4e, reg: %0.4e, cost: %0.4e, ' + \
                'scale: %0.4e'
            print msg % (i, ssqes[i], regs[i], costs[i], current_scale)
            best_reg = regs[i]
            best_ssqe = ssqes[i]
            best_cost = costs[i]
            best_scale = scales[i]

        # go down in scale
        current_scale = np.exp(np.log(current_scale) - parms.search_rate)

    # fill in broken searches
    ind = ssqes == np.inf
    idx = ssqes != np.inf
    ssqes[ind] = np.max(ssqes[idx])

    print 'Old ssqe: %0.4e, reg: %0.4e, total: %0.4e' % \
        (old_valid_ssqe, old_reg, old_cost)
    print 'New ssqe: %0.4e, reg: %0.4e, total: %0.4e, at scale %0.3e' % \
        (best_ssqe, best_reg, best_cost, best_scale)
    
    # update
    psf_model = psf_model - derivatives * best_scale
    psf_model = np.maximum(parms.small, psf_model)
    psf_model /= psf_model.max()

    if parms.plot:
        searchplot(ssqes, regs, scales, parms)

    return psf_model

def validation_score(validation_ssqe, parms, ci=0.68):
    """
    Return the mean validation nll in the given confidence interval
    """
    validation_ssqe = np.sort(validation_ssqe[validation_ssqe < parms.max_ssqe])
    N = validation_ssqe.size
    dlt = (1. - ci) / 2.
    lo = np.ceil(dlt * N)
    hi = np.floor((1. - dlt) * N)
    return np.mean(validation_ssqe[lo:hi])
