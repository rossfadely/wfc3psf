import numpy as np

def evaluate(data, psfs, parms, oldvar=None, return_scaled):
    """
    Fit the data, record the log likelihoods and params, and generate the 
    scaled data and uncertainties.
    """
    if oldvar is None:
        tmpvar = parms.floor + parms.gain * np.abs(data)
    else:
        tmpvar = oldvar

    lnlikes = np.zeros(data.shape[0])
    scaled_data = np.zeros_like(data)
    scaled_data_vars = np.zeros_like(data)
    for i in range(data.shape[0]):

        # fit once under current noise model
        fitparms = fit_single_patch(data[i], psfs[i], tmpvar[i], False)
        model = fitparms[0] * psfs[i] + fitparms[1]

        # fit again under the revised noise model
        tmpvar[i] = parms.floor + parms.gain * np.abs(model)
        fitparms, fituncs = fit_single_patch(data[i], psfs[i], tmpvar[i], True)
        model = fitparms[0] * psfs[i] + fitparms[1]

        # log likelihood
        lnlikes[i] = np.sum(np.log(tmpvar[i]) + (data[i] - model) ** 2. / 
                            tmpvar[i])

        # scale the data and make var
        scaled_data[i] = (data[i] - fitparms[1]) / fitparms[0]
        scaled_data_vars[i] = fitvars[0] * (scaled_data[i] / fitparms[0]) ** 2.
        scaled_data_vars[i] += fitvars[1] / fitparms[0] ** 2.

    return lnlikes, tmpvar, scaled_data, scaled_data_vars

def fit_single_patch(data, psf, var, return_unc):
    """
    Fit the psf model to the data with a constant backround.
    """
    A = np.vstack((psf, np.ones_like(psf))).T

    # fit the data using least squares
    rh = np.dot(A.T, data / var)
    lh = np.linalg.inv(np.dot(A.T, A / var))
    fit_parms = np.dot(lh, rh)
    
    # return uncertainties on the fit if desired
    if return_unc:
        return fit_parms, lh
    else:
        return fit_parms
