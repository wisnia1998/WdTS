import numpy as np
import pandas as pd
import math
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv



##Get data form file
def getData(path):
    file = open(path)
    data = csv.reader(file)
    time, P = [], [],
    for i in data:
        try:
            t, p = i
            time.append(float(t))
            P.append(float(p))
        except:
            pass
        
    return time, P
#############################

##DATA
path_P_H = 'cisnienie_wysokosc.csv'
path_P_T = 'cisnienie_temperatura.csv'
time, P = getData(path_P_H)
p_m = 11.3
h_pietro = 3

##Zad1:
h = (max(P) - min(P))/p_m
ilosc_pieter = math.floor(h/h_pietro)
print('Ciśnienie maksymalne: ' + str(max(P)))
print('Ciśnienie minimalne: ' + str(min(P)))
print('Wysokość: ' + str(round(h,2)) + 'm')
print('ilość pięter: ' + str(ilosc_pieter))
##ciśnienie bezwzgledne
g = 9.78 ##przyspieszenie ziemskie
T = (18 + 459.67) * 5/9 ##dwie ostatnie cyfry indeksu: 18
R = 8.31 ##Stała gazowa
u = 0.0289644 ##masa molowa powietrza
p0 = 101000 ##Ciśnienie odniesienia ( na poziomie morza - w paskalach)
P_h = min(P) ##Ciśnienie na najnizszym pietrze budynku
h_b = np.log(P_h/p0) * (-R * T)/ (u*g) ##przekształcony wzór barometryczny

print('Wysokość bezwzględna podłoża budynku: ' + str(round(h_b,2)) + 'm')

#######################
sos = signal.butter(2, 0.5, 'low', fs = 5, output = 'sos')
P_filt = signal.sosfilt(sos, P)

plt.figure(1)
plt.xlabel('Czas [s]')
plt.ylabel('Ciśnienie [HPa]')
plt.title('Wykres zależności ciśnienia od czasu')
#plt.plot(time[25:],P[25:]) ##bez filtorwania
plt.plot(time[25:],(P_filt[25:])/100,'r')
plt.show()

##Zależność wysokości od czasu
HH = []
for p in P_filt:
    HH.append((101000 - p)/11.3)
plt.xlabel('Czas [s]')
plt.ylabel('Wysokość  [m]')
plt.title('Wykres zależności wysokości od czasu')
plt.plot(time[25:],HH[25:])
################################

##Zad2:
def getData2(path):
    file = open(path)
    data = csv.reader(file, quotechar ='.')
    time, P, T = [], [], [],
    for i in data:
        try:
            t, p, tm = i
            time.append(float(t))
            P.append(float(p))
            T.append(float(tm))
        except:
            pass
        
    return time, P, T

time1 ,P1, T = getData2(path_P_T)

plt.figure(9)
plt.xlabel('Ciśnienie [HPa]')
plt.ylabel('Temperatura [C]')
plt.title('Wykres zależności ciśnienia od temperatury')
P1_HPa = [P / 100 for P in P1]
plt.plot(P1_HPa, T)
plt.show()
#############################################

def func(x, A, B, C, D):
    return A*np.exp(-B*x + C) + D

T = np.array(T)
time1 = np.array(time1)

min_temp_index = np.where(T==np.min(T))

time_d = time1[:min_temp_index[0][0]]
T_d = T[:min_temp_index[0][0]]

T_u = T[min_temp_index[0][0]:]
time_u = time1[min_temp_index[0][0]:] - min_temp_index[0][0]

fit_params_d, _ = curve_fit(func, time_d, T_d)
plt.figure()
plt.title('Schładzanie słoika')
plt.xlabel('Czas [s]')
plt.ylabel('Temperatura [C]')
TZ = plt.plot(time_d,T_d, 'b.', label ='Temperatura zmierzona')
FD = plt.plot(time_d, func(time_d, *fit_params_d), 'r', label = 'Funkcja dopasowania')
plt.text(3000,10, s = 'A = %f\n\
B = %f\n\
C = %f\n\
D = %f\n' %(fit_params_d[0], fit_params_d[1], fit_params_d[2], fit_params_d[3]))
plt.legend()
    
fit_params_u, _ = curve_fit(func, time_u, T_u)
plt.figure()
plt.title('Ogrzewanie słoika')
plt.xlabel('Czas [s]')
plt.ylabel('Temperatura [C]')
plt.plot(time_u,T_u, 'b.', label ='Temperatura zmierzona')
plt.plot(time_u, func(time_u, *fit_params_u), 'r', label = 'Funkcja dopasowania')
plt.text(3000,10, s = 'A = %f\n\
B = %f\n\
C = %f\n\
D = %f\n' %(fit_params_u[0], fit_params_u[1], fit_params_u[2], fit_params_u[3]))
plt.legend()

plt.show()








