# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:34:10 2021

@author: FA
"""

import numpy as np
import matplotlib.pyplot as plt


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


## Definción de constantes
T = 1
sr = 44100    
t = np.linspace(0,T,sr)
freq2 = 10000        
u2 = 0.2          
s2 = 0.05
freq3 = 500
u3 = 0.7
s3 = 0.07
amplitud1 = 2

# Funcion x1
x1 = amplitud1*np.ones_like(t)

#Funcion x2
val1 = np.cos((np.pi)*2*freq2*t)
val2 = np.exp(-((t-u2)**2)/(2*(s2**2)))
x2 = val1*val2

#Funcion 3
val3 = np.sin((np.pi)*2*freq3*t)             
val4 = np.exp((-1)*((t-u3)**2)/(2*(s3**2)))    
x3 = val3*val4

       
#funcion suma
x_final = x1+x2+x3

#Ventana rectangular
M = 2
h = np.ones(M)/M

#Convolucion lineal
conv = np.convolve(x_final,h)
conv = conv[1:-(len(conv)-len(t))]

#Convolucion mediante DFT
y,h_final = convolucion_DFT(x_final,h)
y = y[1:-(len(y)-len(t))]



plt.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=[20, 10])
axs[0].plot(t, x_final)
axs[0].set_title('$x_{final}(t)$')
axs[1].plot(t[:-(len(t)-len(y))],y, color='orange')
axs[1].set_title('convolucion de $x(t)$ con $h(t)$ mediante $DFT$')
axs[2].plot(t[:-(len(t)-len(conv))],conv, color='violet')
axs[2].set_title('Convolucion lineal de $x(t)$ con $h(t)$')
fig.tight_layout()

axs.flat[0].set(ylabel='Amplitud')
axs.flat[1].set(ylabel='Amplitud')
axs.flat[2].set(ylabel='Amplitud', xlabel='Tiempo [s]')

for ax in axs.flat:
    ax.grid()

""" OBSERVACION:
    
-Hay que explicar como se llega a la convolucion lineal mediante la DFT 
por medio de la convolucion circular

- Cambiar el docstring
"""
