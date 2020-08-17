import numpy as np


def cycle(arr, nStep):
    return np.hstack([arr[-nStep:], arr[:-nStep]])


def mix(x, y, frac):
    return (1 - frac) * x + frac * y


def conv_exp(data, dt, tau):
    nTexp = int(5*tau / dt)
    t = dt * np.arange(nTexp)
    exp = np.exp(-t/tau)
    exp /= np.sum(exp)
    nTData = data.shape[0]
    return np.convolve(data, exp)[:nTData]


def two_node_system(nTime, lags, trgFracs, noiseFrac=0.1, crossXY=0, crossYX=0, convDT=None, convTau=None):
    x = np.random.normal(0, 1, nTime)
    y = np.random.normal(0, 1, nTime)
    
    # Add lagged coupling
    y = (1 - np.sum(trgFracs)) * y + np.sum([frac * cycle(x, lag) for frac, lag in zip(trgFracs, lags)], axis=0)
    
    # Add cross-talk
    xMixed = mix(x, y, crossXY)
    yMixed = mix(y, x, crossYX)
    
    # Add convolution
    if convDT is not None:
        xMixed = conv_exp(xMixed, convDT, convTau)
        yMixed = conv_exp(yMixed, convDT, convTau)
        
    # Add observation noise
    xMixed = mix(xMixed, np.random.normal(0, 1, nTime), noiseFrac)
    yMixed = mix(yMixed, np.random.normal(0, 1, nTime), noiseFrac)
    
    return np.array([xMixed, yMixed])