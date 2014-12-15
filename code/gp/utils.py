import numpy as np
import multiprocessing

class FitParms(object):
    """
    Bag of fitting parameters to be passed to functions.
    """
    def __init__(self, patch_shape=(25, 25), Nthreads=4, tol=8, scale=0.1,
                 core_size=5, k=2):

        # be a perfect sqrt
        if (np.sqrt(1. * Nthreads) % 1 != 0):
            print 'I dont like you number of threads, pick a perfect square.'

        self.k = k
        self.buff = tol * scale
        self.Nthreads = Nthreads
        self.core_size = core_size
        self.patch_shape = patch_shape
        self.define_regions(Nthreads, patch_shape)

    def define_regions(self, Nthreads, patch_shape):
        """
        Define the regions of the psf that will be broken up into multiple 
        gps.
        """
        md = (np.array(patch_shape) - 1) / 2 + 0.5
        Nregions = (np.floor(np.sqrt(np.float(Nthreads))).astype(np.int),
                         np.ceil(np.sqrt(np.float(Nthreads))).astype(np.int))
        dlts = 2. * md / Nregions
        self.regions = {}
        self.regions['low'] = []
        self.regions['upp'] = []
        for i in range(Nregions[0]):
            for j in range(Nregions[1]):
                lower = np.array([-md[0] + i * dlts[0], -md[1] + j * dlts[1]])
                upper = lower + dlts
                self.regions['low'].append(lower)
                self.regions['upp'].append(upper)

def run_multi(function, arglist):
    """
    Use multiprocessing to run the specified function.
    """
    pool = multiprocessing.Pool(parms.Nthreads)
    results = list(pool.map(function, [args for args in argslist]))
    pool.close()
    pool.terminate()
    pool.join()
    return results

if __name__ == '__main__':
    f = FitParms(Nthreads = 4)
    print f.regions
