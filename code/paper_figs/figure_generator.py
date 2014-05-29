import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

from matplotlib.colors import LogNorm
from scipy.interpolate import RectBivariateSpline
from matplotlib.patches import ConnectionPatch
from scipy.ndimage.morphology import binary_dilation as grow_mask

np.random.seed(1234)

def fig1(small=1.e-6):
    """
    Figure showing examples of data patches
    """
    Nexamples = 16
    rows = 4
    cols = 4
    rsize = 4
    csize = 4
    shrink = 1.
    ws, hs = 0.0, 0.2
    fs = 24

    # load data and dq
    f = pf.open('../data/region/f160w_25_457_557_457_557_pixels.fits')
    d = f[0].data
    f.close()
    f = pf.open('../data/region/f160w_25_457_557_457_557_dq.fits')
    dq = f[0].data
    f.close()

    # toss data where dq is nonzero in center
    center_pixel = (d.shape[1] - 1 ) / 2
    ind = dq[:, center_pixel] == 0
    d = d[ind]
    dq = dq[ind]

    # sort by maxima
    mxs = d[:, center_pixel]
    ind = np.argsort(mxs)
    mxs = mxs[ind]
    d = d[ind]
    dq = dq[ind]

    # get 7 other random patches
    ind = [0, d.shape[0] - 1]
    count = 0
    while True:
        new = np.random.randint(d.shape[0])
        if new in ind:
            continue
        else:
            ind.append(new)
        if len(ind) == Nexamples:
            break

    # sort and reshape to 2D
    ind = np.array(ind)
    data = d[ind]
    dq = dq[ind]
    mxs = data[:, center_pixel]
    ind = np.argsort(mxs)
    patch_side_size = np.sqrt(data.shape[1]) # assumes square patches
    dq = dq[ind].reshape(Nexamples, patch_side_size, patch_side_size)
    data = data[ind].reshape(Nexamples, patch_side_size, patch_side_size)

    # the figure
    pl.gray()
    f = pl.figure(figsize=(rows * rsize, cols * csize))
    pl.subplots_adjust(wspace=ws, hspace=hs)
    
    axes = [None] * cols
    for i in range(Nexamples):

        # patch pixels with dq != 0
        bad_pix = np.where(dq[i] != 0)
        for j in range(bad_pix[0].size):
            ind = np.zeros_like(dq[i])
            ind[bad_pix[0][j], bad_pix[1][j]] = 1
            idx = grow_mask(ind)
            fix = np.median(data[i][idx])
            data[i][bad_pix[0][j], bad_pix[1][j]] = fix

        mn = data[i].min()
        mx = data[i].max()
        kwargs = {'interpolation':'nearest', 'origin':'lower',
                  'norm':LogNorm(vmin=mn, vmax=mx)}

        ax = pl.subplot(rows, cols, i + 1)
        pl.imshow(data[i], **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        pl.title('[%0.1f, %0.0f]' % (mn, mx), fontsize=fs)

    f.savefig('../plots/paper/fig1.png', bbox_inches='tight')
    pl.close(f)

def fig2():
    """
    Make a plot demonstrating the generation of a psf from
    the pixel-convolved psf model.
    """
    f = pf.open('../psfs/tinytim-pixelconvolved-507-507-5-41.fits')
    model = f[0].data
    f.close()

    extent = 2.5
    center = (-0.25, 0.125)

    # define model grid
    xg = np.linspace(-extent, extent, model.shape[0])
    yg = xg.copy()
    interp_func = RectBivariateSpline(xg, yg, model)

    x = np.array([-2, -1, 0, 1, 2]) + center[0]
    y = np.array([-2, -1, 0, 1, 2]) + center[1]
    psf = interp_func(x, y)

    x, y = np.meshgrid(x, y)
    f = pl.figure(figsize=(10, 5))

    pl.gray()
    ax1 = pl.subplot(121)
    ax1.imshow(model, interpolation='nearest', origin='lower',
              extent=(-extent, extent, -extent, extent),
              norm=LogNorm(vmin=model.min(), vmax=model.max()))
    ax1.plot(x, y, 's', mec='r', mfc='none', mew=2)
    [i.set_linewidth(0.0) for i in ax1.spines.itervalues()]

    pl.xlim(-2.5, 2.5)
    pl.ylim(-2.5, 2.5)
    ax2 = pl.subplot(122)
    ax2.imshow(psf, interpolation='nearest', origin='lower',
               extent=(-extent, extent, -extent, extent),
               norm=LogNorm(vmin=model.min(), vmax=model.max()))

    ax2.set_xticks([-2, -1, 0, 1, 2])
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_xticklabels(['%0.3f' % v for v in x[0]])
    ax2.set_yticklabels(['%0.3f' % v for v in y[:, 0]])
    [i.set_linewidth(0.0) for i in ax2.spines.itervalues()]

    coordsA, coordsB = "data", "data"
    pixels = np.array([[0.0, 0.0], [2., 2.], [-1., -1.]])
    locs = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]])
    rads = [0.15, 0.25, -0.25]
    for i, p in enumerate(pixels):
        xy1 = p + center
        xy2 = p + locs[i]
        con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA=coordsA,
                              coordsB=coordsB, axesA=ax2, axesB=ax1,
                              arrowstyle="<-, head_length=1.2, head_width=0.8", 
                              shrinkB=5,
                              connectionstyle='arc3, rad=%s' % rads[i],
                              color='r', lw=2)
        ax2.add_artist(con)
        ax2.plot(p[0], p[1], 's', mfc='none', mec='r', mew=2, ms=50)

    f.savefig('../plots/paper/fig2.png')
    pl.close(f)


if __name__ == '__main__':

    funcs = [fig2]
    for f in funcs:
        f()
