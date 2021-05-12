# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:44:05 2021

@author: Ezequiel
"""

import math
import matplotlib.pyplot as plt
import numpy as np

## Definción de constantes
T = 1
sr = 44100    
t = np.arange(0,T,1/sr) 
freq3 = 10008.37
u3 = 0.7
s3 = 0.07
muestras = T*sr



def f3(t):
    """
    Función que crea la funcion X3

    Parameters
    ----------
    t : Dominio de la función a crear

    Returns
    -------
    x3 : Seno modulado en una exponencial

    """
    x3 = []
    for i in t:
        val1 = math.sin((math.pi)*2*freq3*i)             
        val2 = math.e**((-1)*((i-u3)**2)/(2*(s3**2)))    
        val = val1*val2
        x3.append(val)  
    return x3

def señalAdB(x,t):
    """
    Funcion que pasa a dB la amplitud de la señal que ingresa

    Parameters
    ----------
    x : Señal en cuestion
    t : Dominio de la señal

    Returns
    -------
    xdB : Señal con amplitud en dB

    """
    xdB = []
    counter = 0
    for i in t:
        val = 20*np.log10(x[counter]/np.max(x))
        counter += 1
        xdB.append(val)
    return xdB


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
    return indiceMaximo + ((alfa-gama)/(2*(alfa-2*beta+gama)))
            
        

x3 = f3(t)

transFourierXfinal = abs(np.fft.fft(x3))

transFourierXfinaldB = señalAdB(transFourierXfinal, t)

freq = np.fft.fftfreq(sr,1/sr)   

maximoInterpolado = interpolacionCuadratica(transFourierXfinaldB)


print('\nMaximum value:',             # Determino el maximo y índice del mismo
      np.max(transFourierXfinaldB),
      "At index:",
      transFourierXfinaldB.index(np.max(transFourierXfinaldB))) 

print('\nMaximum value:',             # Determino el maximo y índice del con interpolacion cuadratica
      np.max(transFourierXfinaldB),
      "At index (interpolate):",
      round(maximoInterpolado,2)) 



fig1=plt.figure()
plt.semilogx( freq , transFourierXfinaldB )
#plt.axis([0,len(transFourierXfinal),-100,4000])
plt.title('')
plt.xlabel('Frequency [Hz]',fontsize=12)
plt.ylabel('Amplitud')
plt.legend()
plt.show()









