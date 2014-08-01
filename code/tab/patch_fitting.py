import numpy as np

from generation import render_psfs
from scipy.ndimage.morphology import binary_dilation as grow_mask

class PatchFitter(object):
    """
    Routines for fitting individual patches of centered, 'stellar'
    data.
    """
    def __init__(self, data, dq, shifts, psf_model, parms, clip,
                 kind, noise_model=None):
        assert kind in ['noise', 'nll']
        psfs = render_psfs(psf_model, shifts, parms.patch_shape, parms.psf_grid,
                           parms.k)
        if kind == 'noise':
            return self.get_noise_model(data, dq, shifts, psfs, parms, clip)
        if kind == 'nll':
            return self.get_nll(data, dq, shifts, psfs, parms, clip,
                                noise_models)

    def get_nll(data, dq, shifts, psfs, parms, clip, noise_models,
                use_current_noise_model):
        """
        Fit the patches and get the negative log likelihood.
        """
        fit_parms, fit_vars, fit_masks = self.fit_patches(data, dq, parms, clip,
                                                          psfs, noise_models)
        scaled_psfs, bkgs = self.construct_model_parts(fit_parms, psfs)
        models = scaled_psfs + bkgs
        if use_current_noise_model:
            return nll(data, models, fit_masks, parms), models
        else:
            return nll(data, models, fit_masks, parms, noise_models), models

    def get_noise_model(self, data, dq, shifts, psfs, parms, clip):
        """
        Construct the noise model for patches, given the psf model.
        """
        cur_nll, models = get_nll(data, dq, shifts, psfs, parms, clips, None, True)
        noise_models = parms.floor + parms.gain * np.abs(models)
        while dlt_nll < parms.nll_tol:
            new_nll, models = get_nll(data, dq, shifts, psfs, parms, clips,
                                      noise_models, False)
            dlt_nll = new_nll - cur_nll
            if dlt_nll < 0.0:
                cur_nll = new_nll
                noise_models = parms.floor + parms.gain * np.abs(models)

        return noise_models

    def construct_model_parts(self, fit_parms, psfs):
        """
        Make the backgrounds for the patch models.
        """
        if fit_parms.shape[1] == 1:
            scaled_psfs = fit_parms[:, None] * psfs
            bkgs = np.zeros_like(psf_models)
        else:
            scaled_psfs = fit_parms[:, 0, None] * psfs
            if fit_parms.shape[1] == 2:
                bkgs = np.zeros_like(psf_models) + fit_parms[:, -1, None]
            else:
                assert False, 'need to add linear bkg building'
        return scaled_psfs, bkgs

    def fit_patches(self, data, dq, parms, clip, psfs, noise_models):
        """
        Fit the data given the psf_model
        """
        flags = np.zeros_like(data, dtype=np.int)
        fit_vars = np.zeros_like(data)
        fit_masks = np.ones_like(data, dtype=np.bool)
        for i in range(data.shape[0]):
            fp, fv, ind = self.fit_single_patch(data[i], dq[i], psfs[i],
                                                noise_models[i])
            if i == 0:
                fit_parms = np.zeros(data.shape[0], fp.shape[1])

            if clip:
                for j in range(parms.clip_iter):

                    # define model and clip variance
                    scaled_psf, bkg = self.construct_model_parts(fp[None, :], psfs)
                    model = scaled_psf + bkg
                    clip_var = parms.floor + parms.gain * np.abs(model)
                    clip_var += parms.q * scaled_psf ** 2.
                    
                    # sigma clip
                    chi = np.zeros_like(data)
                    chi[ind] = np.abs(data[ind] - model[ind]) / np.sqrt(clip_var[ind])
                    condition = chi - parms.clip_tol
                    condition = (condition > 0).reshape(parms.patch_shape)

                    # redefine mask, grow and add to dq mask.
                    ind = 1 - ind.reshape(parms.patch_shape)
                    idx = grow_mask(condition)
                    flags[i] = ind.copy()
                    flags[i][idx] = 3
                    flags[i][condition] = 2
                    ind = np.ravel((ind == 0) & (idx == 0))

                    # refit
                    fp, fv, ind = self.fit_single_patch(data[i], dq[i], psfs[i],
                                                        noise_models[i], ind=ind)

            fit_vars[i] = fv
            fit_masks[i] = ind
            fit_parms[i] = fp

        return fit_parms, fit_vars, fit_masks

    def fit_single_patch(self, datum, dq, psf, parms, var=None, ind=None):
        """
        Fit a single patch using least squares.
        """
        gain = parms.gain
        floor = parms.floor
        background = parms.background

        if var == None:
            var = np.ones_like(data)
        if ind == None:
            # in dq array, zero = good, else = bad
            ind = dq == 0

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

        # fit the data using least squares
        ATICY = np.dot(A[ind, :].T, datum[ind] / var[ind])
        try:
            ATICA = np.linalg.inv(np.dot(A[ind, :].T, A[ind, :] /
                                         var[ind, None]))
            fit_parms = np.dot(ATICA, ATICY)
        except:
            fit_parms = np.zeros(A.shape[1])

        return fit_parms, np.diagonal(ATICA), ind

    def nll(self, data, models, fit_masks, parms, noise_models=None):
        """
        Compute the negative log likelihood of the patches using the specified masks.
        """
        nll = np.zeros_like(data)
        if noise_models == None:
            var = parms.floor + parms.gain * np.abs(model)
        else:
            var = noise_models

        nll[fit_masks] = 0.5 * np.sqrt(var[fit_masks])
        nll += 0.5 * (data[fit_masks] - models[fit_masks]) ** 2. / var[fit_masks]

        # assign max nll to paches where fit failed.
        ind = models == 0
        nll[ind] = parms.max_nll

        return nll
