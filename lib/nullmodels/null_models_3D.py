import numpy as np


def bernoulli(n, p):
    return (np.random.uniform(0, 1, n) < p).astype(int)


#########################################
# Discrete models
#########################################

def _add_discr_noise(n, x, y, z, alphaX=0.5, alphaY=0.5, alphaZ=0.5):
    aX = bernoulli(n, alphaX)
    aY = bernoulli(n, alphaY)
    aZ = bernoulli(n, alphaZ)
    xNew = (1 - aX) * x + aX * bernoulli(n, 0.5)
    yNew = (1 - aY) * y + aY * bernoulli(n, 0.5)
    zNew = (1 - aZ) * z + aZ * bernoulli(n, 0.5)
    return xNew, yNew, zNew


def discr_red_noisy(nSample, alphaX=0.5, alphaY=0.5, alphaZ=0.5):
    t = bernoulli(nSample, 0.5)
    return _add_discr_noise(nSample, t, t, t, alphaX=alphaX, alphaY=alphaY, alphaZ=alphaZ)


def discr_unq_noisy(nSample, alphaX=0.5, alphaY=0.5, alphaZ=0.5):
    t = bernoulli(nSample, 0.5)
    return _add_discr_noise(nSample, t, 0, t, alphaX=alphaX, alphaY=alphaY, alphaZ=alphaZ)


def discr_syn_noisy(nSample, alphaX=0.5, alphaY=0.5, alphaZ=0.5):
    x = bernoulli(nSample, 0.5)
    y = bernoulli(nSample, 0.5)
    z = np.logical_xor(x, y)
    return _add_discr_noise(nSample, x, y, z, alphaX=alphaX, alphaY=alphaY, alphaZ=alphaZ)


def discr_method_dict():
    return {
        'discr_red': discr_red_noisy,
        'discr_unq': discr_unq_noisy,
        'discr_syn': discr_syn_noisy,
    }


#########################################
# Continuous models
#########################################

def _add_cont_noise(n, x, y, z, sigX=1, sigY=1, sigZ=1):
    xNew = x + np.random.normal(0, sigX, n)
    yNew = y + np.random.normal(0, sigY, n)
    zNew = z + np.random.normal(0, sigZ, n)
    return xNew, yNew, zNew


def cont_red_noisy(n=1000, sigX=1, sigY=1, sigZ=1):
    t = np.random.normal(0, 1, n)
    return _add_cont_noise(n, t, t, t, sigX=sigX, sigY=sigY, sigZ=sigZ)


def cont_unq_noisy(n=1000, sigX=1, sigY=1, sigZ=1):
    t = np.random.normal(0, 1, n)
    return _add_cont_noise(n, t, 0, t, sigX=sigX, sigY=sigY, sigZ=sigZ)


def cont_xor_noisy(n=1000, sigX=1, sigY=1, sigZ=1):
    x0 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    return _add_cont_noise(n, x0, y0, x0 * y0, sigX=sigX, sigY=sigY, sigZ=sigZ)


def cont_method_dict():
    return {
        'cont_red': cont_red_noisy,
        'cont_unq': cont_unq_noisy,
        'cont_syn': cont_xor_noisy,
    }
