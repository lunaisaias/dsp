# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:49:18 2021

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
freq3 = 500
u3 = 0.7
s3 = 0.07
amplitud1 = 2


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
    
    Xfinal = []
    counter = 0
    for i in t:
        val = x1[counter] + x2[counter]+ x3[counter]
        counter += 1
        Xfinal.append(val)
    return Xfinal

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
    h = np.ones(M)/(M+1)                       # Creo respuesta al impulso de ventana M
    xFiltrada = np.convolve(h,x,'valid')       # Realizo convolución
    for n in range(M-1):                       # Agrego el ultimo valor M veces para que la señal resultante me quede del mismo largo que la señal original
        xFiltrada = np.append(xFiltrada,xFiltrada[-1])
    return xFiltrada

def calculadoraDeM(x,frecuencia,atenuacion):
    M = 1
    valorEnFrecuencia = x[frecuencia]
    valorFinal = atenuacion*valorEnFrecuencia
    while atenuacion*valorEnFrecuencia > x[frecuencia]:
        x = convImpulseResponse(x,M)
        M += 1
    print(f'El valor de la señal en {frecuencia} Hz es: {round(valorEnFrecuencia,3)} dB,',
          f'con una atenuación de {atenuacion}:',
          round(valorFinal,3),f'dB. Aplicando el filtro pedido con M = {M-1},',
          f'el valor en {frecuencia} Hz resulta:',round(x[frecuencia],2),'dB.')
    return M
        


x = funcionSuma(t)

transFourierX = abs(np.fft.fft(x))

transFourierXdB = señalAdB(transFourierX, t)

freq = np.fft.fftfreq(sr,1/sr) 

xFiltrada = convImpulseResponse(transFourierXdB,2)

M = calculadoraDeM(transFourierXdB,10000,0.07)


fig1=plt.figure()
plt.semilogx( freq , xFiltrada )
#plt.axis([0,len(transFourierXfinal),-100,4000])
plt.title('')
plt.xlabel('Frequency [Hz]',fontsize=12)
plt.ylabel('Amplitud')
plt.legend()
plt.show()

