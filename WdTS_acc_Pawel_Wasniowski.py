import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv

##Get data form file
def getData(path):
    file = open(path)
    data = csv.reader(file)
    time, acc = [], [],
    for i in data:
        try:
            t, a = i
            time.append(float(t))
            acc.append(float(a))
        except:
            pass
        
    return time, acc
#############################

##Integ - calculate rectangle integral
def integral(x,y):
    S = np.zeros_like(y)
    S[0] = (y[0] + y[1])/2*(x[1]-x[0])#calculate first element of integral
    for i in range(1, len(y)-1):
        S[i] = S[i-1] + (y[i] + y[i+1])/2 * (x[i+1] - x[i])
    return S
####################

##Data
dane_s = 'dane_sprezyna.csv'
time, acc_raw = getData(dane_s)
g = 9.81 #gravity constant
lsb_g = np.power(2,15) #number of bits per 1g acceleration
offset = np.mean(acc_raw)
acc = (acc_raw - offset) * g / lsb_g
##################################

##Acceleration
def accel(t,a):
    plt.figure()
    plt.title('Acceleration/time')
    plt.xlabel('Time[s]')
    plt.ylabel('Acceleration[m/s^2]')
    plt.plot(t,a)
############################
##Velocity
def vel(t, acc):
    v = integral(t, acc)
    
    sos = signal.butter(2, [1,3], 'bp', fs = 100, output = 'sos')
    v_filt = signal.sosfilt(sos, v)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Velocity/time char')
    plt.xlabel('Time[s]')
    plt.ylabel('Velocity[m/s]')
    plt.plot(t,v)
    
    plt.subplot(2,1,2)
    plt.title('Filtered velocity/time char')
    plt.xlabel('Time[s]')
    plt.ylabel('Velocity[m/s]')
    plt.plot(t,v_filt)

    return v, v_filt
###############

##Position
def position(t, v):
    x = integral(t, v)

    sos = signal.butter(2, [1,3], 'bp', fs = 100, output = 'sos')
    x_filt = signal.sosfilt(sos, x)

    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Position/time')
    plt.xlabel('Time[s]')
    plt.ylabel('Position[cm]')
    plt.plot(t,x*100)

    plt.subplot(2,1,2)
    plt.title('Filtered position/time')
    plt.xlabel('Time[s]')
    plt.ylabel('Position[cm]')
    plt.plot(t,x_filt*100)
    
    return x, x_filt

##########################

#fft
def fft_x(c):

    fft = np.fft.fft(c)
    freq = np.fft.fftfreq(c.shape[-1])*1000
    freq = freq[:250]
    amp = abs(fft[:int(len(freq) / 2)])
    amp = amp[:250]
    print('freq of max spectrum band: ' + str(freq[67].round(2)) + 'Hz')
    plt.figure()
    plt.title('FFT transformation')
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplituda')
    plt.plot(freq[:int(len(freq) / 2)], amp)

######################################

##Dopasowanie
def func(x, a,b,c,d):
    return a * np.sin(b*x + c) + d

def fit(t, x):
    fit_params, covariance_matrix = curve_fit(func, t, x, p0=[0.0013, 13.4, 0 , 0 ])
    print('a(amplituda) = ' + str(fit_params[0]))
    print('b(częstotliwość) = ' + str(fit_params[1]))
    print('c(przesunięcie fazowe) = ' + str(fit_params[2]))
    print('d(offset) = ' + str(fit_params[3]))
    print('Fitting function: {}*sin({}*x+{})+{}'.format
          (fit_params[0].round(7),fit_params[1].round(3),fit_params[2].round(3),fit_params[3].round(8)))
############################

accel(time,acc)
v , v_filt = vel(time, acc)
x,x_filt=position(time,v_filt)
fft_x(x_filt)
fit(time, x_filt)

plt.show()

    
  

