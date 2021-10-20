import numpy as np


def _add_noise(n, x, y, z, sigX=1, sigY=1, sigZ=1):
    xNew = x + np.random.normal(0, sigX, n)
    yNew = y + np.random.normal(0, sigY, n)
    zNew = z + np.random.normal(0, sigZ, n)
    return xNew, yNew, zNew


def gen_data_red_noisy(n=1000, sigX=1, sigY=1, sigZ=1):
    t = np.random.normal(0, 1, n)
    return _add_noise(n, t, t, t, sigX=sigX, sigY=sigY, sigZ=sigZ)


def gen_data_unq_noisy(n=1000, sigX=1, sigY=1, sigZ=1):
    t = np.random.normal(0, 1, n)
    return _add_noise(n, t, 0, t, sigX=sigX, sigY=sigY, sigZ=sigZ)


def gen_data_xor_noisy(n=1000, sigX=1, sigY=1, sigZ=1):
    x0 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    return _add_noise(n, x0, y0, x0 * y0, sigX=sigX, sigY=sigY, sigZ=sigZ)
