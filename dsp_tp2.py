# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

def blackman(M, a0=0.42, a1=0.5, a2=0.08):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    a0 : TYPE, optional
        DESCRIPTION. The default is 0.42.
    a1 : TYPE, optional
        DESCRIPTION. The default is 0.5.
    a2 : TYPE, optional
        DESCRIPTION. The default is 0.08.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    '''
    n = np.arange(M)
    blackman = a0+a1*np.cos(2*np.pi*n/(M-1))+a2*np.cos(4*np.pi*n/(M-1))
    
    return blackman



def fft(x,t,Nfft):
    '''
    Approximate the Fourier transform of a finite duration signal 
    using scipy.signal.freqz()
    
    Parameters
    ----------
    x : input signal array
    t : time array used to create x(t)
    Nfft : the number of frdquency domain points used to 
           approximate X(f) on the interval [fs/2,fs/2], where
           fs = 1/Dt. Dt being the time spacing in array t
    
    Returns
    -------
    f : frequency axis array in Hz
    X : the Fourier transform approximation (complex)
    
    Notes
    -----
    The output time axis starts at the sum of the starting values in x1 and x2 
    and ends at the sum of the two ending values in x1 and x2. The default 
    extents of ('f','f') are used for signals that are active (have support) 
    on or within n1 and n2 respectively. A right-sided signal such as 
    :math:`a^n*u[n]` is semi-infinite, so it has extent 'r' and the
    convolution output will be truncated to display only the valid results.

    
    '''
    fs = 1/(t[1] - t[0])
    t0 = (t[-1]+t[0])/2 # time delay at center
    N0 = len(t)/2 # FFT center in samples
    f = np.arange(-1./2,1./2,1./Nfft)
    w, X = signal.freqz(x,1,2*np.pi*f)
    X /= fs # account for dt = 1/fs in integral
    X *= np.exp(-1j*2*np.pi*f*fs*t0)# time interval correction
    X *= np.exp(1j*2*np.pi*f*N0)# FFT time interval is [0,Nfft-1]
    F = f*fs
    
    return F, X

def hann(M):

    n = np.arange(M)
    hann = 0.5 - 0.5*np.cos((2*np.pi*n)/(M-1))
    
    return hann

def interpolacionCuadratica(x):
    """
    Funcion que realiza interpolación cuadrática para encontrar una
    mejor aproximación del índice de un máximo

    Parameters
    ----------
    x : Señal de entrada

    Returns
    -------
    Devuelve valor de índice maximo interpolado

    """
    indiceMaximo = x.index(np.max(x))
    beta = abs(x[indiceMaximo])
    alfa = abs(x[indiceMaximo-1])
    gama = abs(x[indiceMaximo+1])
    maximo= indiceMaximo + ((alfa-gama)/(2*(alfa-2*beta+gama)))
    
    return maximo