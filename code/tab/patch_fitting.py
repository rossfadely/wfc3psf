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
                                noise_model)

    def fit_patches(self, data, dq, parms, clip, psfs):
        """
        Fit the data given the psf_model
        """
        fit_vars = np.zeros_like(data)
        fit_masks = np.ones_like(data, dtype=np.bool)
        for i in range(data.shape[0]):
            fp, fv, fm = self.fit_single_patch()
            if i == 0:
                fit_parms = np.zeros(data.shape[0], fp.shape[1])

            if clip:
                for j in range(parms.clip_iter):
                    assert False, 'need to add clipping.'

            fit_vars[i] = fv
            fit_masks[i] = fm
            fit_parms[i] = fp

        return fit_parms, fit_vars, fit_masks

    def get_noise_model(self, data, dq, shifts, psfs, parms, clip):
        """
        Construct the noise model for patches, given the psf model.
        """
        assert 0, 'need to add convergence via nll.'
        fit_parms, fit_vars, fit_masks = self.fit_patches(data, dq, parms, clip,
                                                          psfs)
        scaled_psfs, bkgs = self.construct_model_parts(fit_parms, psfs)
        models = scaled_psfs + bkgs
        return np.abs(models) * parms.gain + parms.floor

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
