# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:34:10 2021

@author: FA
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 



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

# DFT 
DFT_x_final = np.fft.fft(x_final)


plt.figure(1)

f1, t1, Zxx1 = signal.stft(x_final,sr,window = 'blackman',nperseg = 600,noverlap = 500 ,nfft = 5000)
f2, t2, Zxx2 = signal.stft(x_final,sr,window = 'hann',nperseg = 600,noverlap = 500 ,nfft = 5000)
f3, t3, Zxx3 = signal.stft(x_final,sr,window = 'bartlett',nperseg = 700,noverlap = 600 ,nfft = 5000)


plt.figure(figsize=(20,16))
plt.pcolormesh(t1, f1, 20*np.log(abs(Zxx1)))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,17500)
plt.title("Magnitud de STFT utilizando una ventana de Blackman")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.show()

plt.figure(figsize=(20,16))
plt.pcolormesh(t2, f2, 20*np.log(abs(Zxx2)))
plt.ylim(0,17500)
plt.colorbar(format='%+2.0f dB')
plt.title("Magnitud de STFT utilizando una ventana de Hann")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.show()

plt.figure(figsize=(20,16))
plt.pcolormesh(t3, f3, 20*np.log(abs(Zxx3)))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,17500)
plt.title("Magnitud de STFT utilizando una ventana de Bartlett")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.show()

plt.figure(figsize=(20,16))
plt.pcolormesh(t1, f1, np.angle(Zxx1))
plt.colorbar()
plt.title("Fase de STFT utilizando una ventana de Blackman")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.show()

plt.figure(figsize=(20,16))
plt.pcolormesh(t2, f2, np.angle(Zxx2))
plt.colorbar()
plt.title("Fase de STFT utilizando una ventana de Hann")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.show()

plt.figure(figsize=(20,16))
plt.pcolormesh(t3, f3, np.angle(Zxx3))
plt.colorbar()
plt.title("Fase de STFT utilizando una ventana de Bartlett")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.show()




