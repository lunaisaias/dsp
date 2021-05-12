# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:58:07 2021

@author: Ezequiel
"""

import math
import matplotlib.pyplot as plt
import numpy as np

## Definción de constantes
T = 1
sr = 44100    
t = np.arange(0,T,1/sr) 
freq2 = 10000        
u2 = 0.2          
s2 = 0.05
freq3 = 10008.37
u3 = 0.7
s3 = 0.07
amplitud1 = 2
muestras = T*sr
noise = np.random.normal( 0 , 0.01 , muestras)


## Definición de funciones
def f1(t):
    """
    Función que crea la función X1
    
    
    Parameters
    ----------
    t : Dominio de la función a crear.

    Returns
    -------
    x1 : Función de salida: Constante de largo x y amplitud: amplitud1
    """
    x1 = []                   # Creo lista vacía
    for i in t:
        x1.append(amplitud1)     # Agrego un 2 a la lista
    return x1

def f2(t):
    """
    Función que crea la funcion X2

    Parameters
    ----------
    t : Dominio de la función a crear
    Returns
    -------
    x2 : Coseno modulado en una exponencial.
    
    """
    x2 = []
    for i in t:
        val1 = math.cos((math.pi)*2*freq2*i)
        val2 = math.e**(-((i-u2)**2)/(2*(s2**2)))
        val = val1*val2
        x2.append(val)           
    return x2



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

def funcionSuma(t):
    """
    Función que obtiene la señal resultante de la suma de x1, x2 y x3

    Parameters
    ----------
    t : Dominio de la señal resultante

    Returns
    -------
    xresultante : Señal resultante
    """
    # Creo las señales a sumar
    x1 = f1(t)
    x2 = f2(t)
    x3 = f3(t)
    noise = np.random.normal( 0 , 1 , muestras)
    
    Xfinal = []
    counter = 0
    for i in t:
        val = x1[counter] + x2[counter]+ x3[counter] + noise[counter]
        counter += 1
        Xfinal.append(val)
    return Xfinal


def ventanaBlackman(M):
    """
    Función que realiza ventana de blackman de largo M

    Parameters
    ----------
    
    M : Ancho de la ventana

    Returns
    -------
    ventanaBlackman : Ventana de Blackman
    
    """
    ventanaBlackman=[]
    for n in range(M-1):
        ventanaBlackman_value=0.42-0.5*math.cos((2*math.pi*n)/(M-1))+0.08*math.cos((4*math.pi*n)/(M-1))
        ventanaBlackman.append(ventanaBlackman_value)
    ventanaBlackman.append(ventanaBlackman_value)
    return ventanaBlackman

def ventanaRectangular(M):
    """
    Funcion que realiza ventana rectangular de largo M

    Parameters
    ----------
    M : Ancho de la ventana

    Returns
    -------
    ventanaRectangular : ventana rectangular

    """
    ventanaRectangular = []
    for i in M:
        ventanaRectangular.append(1)
    return ventanaRectangular
    
def ventanaHann(M):
    """
    Función que realiza ventana de Hann de largo M

    Parameters
    ----------
    
    M : Ancho de la ventana

    Returns
    -------
    ventanaHann : Ventana de Hann
    
    """
    ventanaHann = []
    for n in range(M-1):
        ventanaHann_value = 0.5 - 0.5*math.cos((2*math.pi*n)/(M-1))
        ventanaHann.append(ventanaHann_value)
    ventanaHann.append(ventanaHann_value)
    return ventanaHann


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

def multiplicadorDeSeñalConVentana(x,ventana):
    """
    Funcion que mutliplica la señal que ingresa con una ventana dada de igual largo

    Parameters
    ----------
    x : Señal en cuestion
    ventana : Ventana a multiplicar

    Returns
    -------
    señalMultiplicada : Señal resultante

    """
    señalMultiplicada = []
    for i in range(0,len(x)):
        señalMultiplicada_value = x[i] * ventana[i]
        señalMultiplicada.append(señalMultiplicada_value)
    return señalMultiplicada
    


ventanaBlackman = ventanaBlackman(muestras)

ventanaHann = ventanaHann(muestras)

ventanaRectangular = ventanaRectangular(muestras)

Xfinal = funcionSuma(t)

señalMultiplicada = multiplicadorDeSeñalConVentana( Xfinal , ventanaBlackman )

transFourierXfinal = abs(np.fft.fft(señalMultiplicada))

transFourierXfinaldB = señalAdB(transFourierXfinal, t)

freq = np.fft.fftfreq(sr,1/sr)   





fig1=plt.figure()
plt.plot( t , ventanaBlackman )
#plt.axis([0,len(transFourierXfinal),-100,4000])
plt.title('')
plt.xlabel('Frequency [Hz]',fontsize=12)
plt.ylabel('Amplitud')
plt.legend()
plt.show()