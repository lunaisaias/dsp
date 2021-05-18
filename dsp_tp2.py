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

def int_cuadratica(x):
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
    indice_maximo = max(x)
    freq_max = np.where(x == indice_maximo)
    freq_max= int(freq_max[0][0])
    
    beta = abs(x[freq_max])
    alfa = abs(x[freq_max-1])
    gamma = abs(x[freq_max+1])
    
    
    maximo = indice_maximo + ((alfa-gamma)/(2*(alfa-2*beta+gamma)))
    
    posicion_maxima =  ((alfa-gamma)/(2*(alfa-2*beta+gamma)))
    
    frecuencia_de_interpolacion = freq_max + posicion_maxima

    amplitud_maxima = beta- ((1/4)*(alfa - gamma)*posicion_maxima)
    
   
    return amplitud_maxima, frecuencia_de_interpolacion


def derivada_finita(x, fs = 44100):

    df = []

    for i in range (0, len(x)-1):
        der = ( x[i+1] - x[i] ) / ( 1/fs )
        df.append(der)
        
  
    return df
    
     

def convolucion_DFT(signal1,signal2):
    
    '''
    Scipy implementation of a moving average filter with a convolutional method.
    The resulting filtered sample is calculated with a convolution between the
    original signal and a rectangular pulse with length 'M'.

    Parameters
    ----------
    signal1 : ndarray
        Input signal.
    
    signal2 : ndarray
        Input signal.
        

    Returns
    -------
    y : ndarray
        Output convolution.

    '''
    x = np.copy(signal1)
    h = np.copy(signal2)
    
    N = len(x) + len(h) - 1        
    
    zeros_1 = np.zeros(N - len(x))
    x = np.hstack((x, zeros_1))
    
    
    zeros_2 = np.zeros(N - len(h))
    h = np.hstack((h, zeros_2))

    x_DFT = np.fft.fft(x)
    h_DFT = np.fft.fft(h)
    
    y_DFT = x_DFT * h_DFT
    
    y = np.fft.ifft(y_DFT)
    
    return abs(y),h

def convImpulseResponse(x,M):
    """
    Funcion que realiza la convolucion entre una entrada y una Respuesta al impulso de 
    ventana M. El modo utilizado para la convolucion es 'valid', para no tener valores
    anómalos en los extremos de la señal resultante

    Parameters
    ----------
    x : TFuncion de entrada
    M : Tamaño de la ventana

    Returns
    -------
    xFiltrada : Funcion convolucionada

    """
    h = np.ones(M)/M
    y = np.convolve(x,h)
    
    return y

def calculadoraDeM(x,frecuencia,atenuacion):
    
    xoriginal = x[frecuencia]
    valorFinal = atenuacion+xoriginal
    
    M = 5
    x = convImpulseResponse(x,M)

    while atenuacion<x[frecuencia]-valorFinal and range(3):
        M += 1
        xfiltrado = convImpulseResponse(x,M)

    return xfiltrado,M,xoriginal

