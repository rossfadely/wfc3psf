import numpy as np

def get_grids(patch_shape, psf_shape, core_shape=None):
    """
    Return grid definitions used during fitting
    """
    # psf_grid defs
    xg = np.linspace(-0.5 * patch_shape[0], 0.5 * patch_shape[0],
                      psf_shape[0])
    yg = np.linspace(-0.5 * patch_shape[1], 0.5 * patch_shape[1],
                      psf_shape[1])
    psf_grid = (xg, yg)

    # define patch_grid
    if core_shape is not None:
        shape = core_shape
    else:
        shape = patch_shape
    xsize = (shape[0] - 1) / 2.
    ysize = (shape[1] - 1) / 2.
    xp, yp = np.meshgrid(np.linspace(-xsize, xsize, shape[0]),
                         np.linspace(-ysize, ysize, shape[1]))
    patch_grid = (xp, yp)

    return psf_grid, patch_grid
