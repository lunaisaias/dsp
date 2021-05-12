# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:11:45 2021

@author: Ezequiel
"""


import math
import matplotlib.pyplot as plt
import numpy as np

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
val3 = np.sin((math.pi)*2*freq3*t)             
val4 = np.exp((-1)*((t-u3)**2)/(2*(s3**2)))    
x3 = val3*val4

       
#funcion suma
x_final = x1+x2+x3

#funciondB
x_dB = 20*np.log(x_final/abs(np.max(x_final)))


plt.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[20, 10])
axs[0, 0].plot(t, x1)
axs[0, 0].set_title('$x_1(t)$')
axs[0, 1].plot(t, x2, color='orange')
axs[0, 1].set_title('$x_2(t)$')
axs[1, 0].plot(t, x3, color='green')
axs[1, 0].set_title('$x_3(t)$')
axs[1, 1].plot(t, x_final, color='red')
axs[1, 1].set_title('$x(t) = x_1(t) + x_2(t) + x_3(t)$')
fig.tight_layout()

axs.flat[0].set(ylabel='Amplitud')
axs.flat[2].set(ylabel='Amplitud', xlabel='Tiempo [s]')
axs.flat[3].set(xlabel='Tiempo [s]')

for ax in axs.flat:
    ax.grid()


    
## LLamados a funciones
                                      # Creo y sumo las funciones
dft_x_final = abs(np.fft.fft(x_final))                 # Le aplico la transformada
dft_x_dB = 20*np.log(dft_x_final/abs(np.max(dft_x_final)))      # Paso la transformada a dB
freq = np.fft.fftfreq(sr,1/sr)                               # Creo escala de frecuencia




# ## Ploteo para probar

plt.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=[20, 10])
axs[0].plot(freq, dft_x_final[0,4000])
axs[0].set_title('DFT $x_final(t)$')
axs[1].semilogx(freq, dft_x_dB[0,4000], color='orange')
axs[1].set_title('DTF dB $x_final(t)$')
fig.tight_layout()

axs.flat[0].set(ylabel='Amplitud')
axs.flat[1].set(ylabel='Amplitud', xlabel='Frecuencia [Hz]')

for ax in axs.flat:
    ax.grid()