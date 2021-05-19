# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

def blackman(M, a0=0.42, a1=0.5, a2=0.08):
    '''
    Funcion que crea una ventana de Blackman.
    

    Parameters
    ----------
    M : int
        Ancho de la ventana
    a0 : float
         Coeficiente blackman. Default = 0.42.
    a1 : float
         Coeficiente blackman. Default = 0.5.
    a2 : float
         Coeficiente blackman. Default = 0.08.

    Returns
    -------
    blackman : ndarray
               Devuelve la ventana de blackman de M muestras.
            

    '''
    n = np.arange(M)
    blackman = a0-a1*np.cos(2*np.pi*n/(M-1))+a2*np.cos(4*np.pi*n/(M-1))
    
    return blackman





def hann(M):
    '''
    Funcion que crea una ventana de Hann.
    
    Parameters
    ----------
    M : int
        Ancho de la ventana

    Returns
    -------
    hann : ndarray
           Devuelve una ventana de Hann de M muestras.
    
    '''

    n = np.arange(M)
    hann = 0.5 - 0.5*np.cos((2*np.pi*n)/(M-1))
    
    return hann

def int_cuadratica(x):
    """
    Funcion que realiza interpolacion cuadratica para encontrar una
    mejor aproximacion del índice de un maximo.

    Parameters
    ----------
    x : ndarray
        Senial de entrada

    Returns
    -------
        amplitud_maxima : float
            Amplitud maxima de la señal segun metodo de interpolacion
        
        frecuencia de interpolacion : float
            Frecuencia en la que se encuentra la amplitud_maxima

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
    """
    Funcion que realiza la derivada finita de una funcion.

    Parameters
    ----------
    x : ndarray
        Senial de entrada

    Returns
    -------
       df : ndarray
           Derivada por metodo finito forward

    """
    df = []

    for i in range (0, len(x)-1):
        der = ( x[i+1] - x[i] ) / ( 1/fs )
        df.append(der)
        
  
    return df
    
     

def convolucion_DFT(signal1,signal2):
    
    '''
    

    Parameters
    ----------
    signal1 : ndarray
        Senial a convolucionar.
    
    signal2 : ndarray
        Respuesta al impulso a convolucionar.
        

    Returns
    -------
    y : ndarray
        Parte real de la convolucion lineal de signal1*signal2.
        
    h : ndarray
        Respuesta al impulso con zero padding.
        
    '''
    
    #Copia las señales
    x = np.copy(signal1)
    h = np.copy(signal2)
    
    #calculo del largo de la convolucion lineal
    N = len(x) + len(h) - 1        
    
    #genera un vector de zeros a agregar a la senial 1 
    zeros_1 = np.zeros(N - len(x))
    x = np.hstack((x, zeros_1))
    
    #genera un vector de zeros a agregar a la senial 2 
    zeros_2 = np.zeros(N - len(h))
    h = np.hstack((h, zeros_2))
    
    #Calculos de DFT
    x_DFT = np.fft.fft(x)
    h_DFT = np.fft.fft(h)
    
    #Producto de las dft
    y_DFT = x_DFT * h_DFT
    
    #Transformada inversa del producto(propiedad de DFT)
    y = np.fft.ifft(y_DFT)
    
    return np.real(y),h

def convImpulseResponse(x,M):
    """
    Funcion que realiza la convolucion entre una entrada y una Respuesta al impulso de 
    ventana M. El modo utilizado para la convolucion es 'valid', para no tener valores
    anómalos en los extremos de la señal resultante.

    Parameters
    ----------
    x : ndarray
        TFuncion de entrada
        
    M : int
        Tamaño de la ventana

    Returns
    -------
    y: ndarray
        Funcion convolucionada

    """
    h = np.ones(int(M))/int(M)
    y = np.convolve(x,h)
    
    return y

def calculadoraDeM(x,frecuencia,atenuacion):
    
    """
    Funcion que calcula el valor de una ventana, en funcion de la atenuacion requerida
    en cierto componete del espectro de la senial.

    Parameters
    ----------
    x : ndarray
        Funcion de entrada
        
    M : int
        Tamanio de la ventana
        
    atenuacion: int
        Atenuacion requerida en dB

    Returns
    -------
    M : int 
        Tamanio de la ventana calculada
    
    x1 : ndarray
        Señial de entrada atenuada por la ventana de tamano M(final).
        
    valorEnFrecuencia: float
        Valor en amplitud del punto deseado sin atenuar.
    
    """
    
    valorEnFrecuencia = abs(20*np.log10(x[frecuencia]))
    
    M = 1
    x1 = convImpulseResponse(x,M)
    x1 = abs(x1)
    x1_db= 20*np.log10(x1)
    

    while range(atenuacion-1,atenuacion,100) > valorEnFrecuencia - x1[frecuencia]:
        M += 1
        x1 = 20*np.log10(abs(convImpulseResponse(x,M)))
       
        
    return x1,M,valorEnFrecuencia
