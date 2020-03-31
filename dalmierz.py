import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

odległość = []
czas_impulsu = []
odczyt_VL53L1X = []
odczyt_HC_SR04 = []
temperatura = []
dalmierz_pomiar_1 = pd.read_excel(r'E:\6 semestr\WdTS\dalmierz\dalmierz_pomiar_1.xlsx')
dalmierz_pomiar_2 = pd.read_excel(r'E:\6 semestr\WdTS\dalmierz\dalmierz_pomiar_2.xlsx')

def odczyt(data):
    odległość = []
    czas_impulsu = []
    odczyt_VL53L1X = []
    odczyt_HC_SR04 = []
    temperatura = []
    for x in data['odległość']:
        odległość.append(float(x))
    for x in data['czas impulsu']:
        czas_impulsu.append(x)
    for x in data['odczyt VL53L1X']:
        odczyt_VL53L1X.append(x)
    for x in data['odczyt HC-SR04']:
        odczyt_HC_SR04.append(x)
    for x in data['temperatura']:
        temperatura.append(x)
    return odległość, czas_impulsu, odczyt_VL53L1X, odczyt_HC_SR04, temperatura

def wykres_odl_czas(x,y,xlabel,ylabel, title):
    plt.plot(x, y,'ro-',markerfacecolor='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.show()


odległość, czas_impulsu ,odczyt_VL53L1X,odczyt_HC_SR04,_ = odczyt(dalmierz_pomiar_1)    
wykres_odl_czas(odczyt_VL53L1X,czas_impulsu, 'odległość [mm]', 'czas [ms]','Czujnik ultradzwiękowy\npomiar_1')# "Czujnik ultradzwiękowy\npomiar_1")

odległość, czas_impulsu ,odczyt_VL53L1X,odczyt_HC_SR04,_ = odczyt(dalmierz_pomiar_2)    
wykres_odl_czas(odczyt_VL53L1X,czas_impulsu, 'odległość [mm]', 'czas [ms]','Czujnik ultradzwiękowy\npomiar_2')# 'Czujnik ultradzwiękowy\npomiar_2')

def sensitivity(d1,t):
    d1 = np.array(d1).reshape(-1, 1)
    t = np.array(t).reshape(-1, 1)
    sensor_s = LinearRegression()
    sensor_s.fit(d1, t)
    print('Sensitivity: ' + str(sensor_s.coef_))
    return sensor_s.coef_ ,sensor_s.intercept_


_,czas_impulsu,odczyt_VL53L1X, odczyt_HC_SR04,_ = odczyt(dalmierz_pomiar_1)
a,b=sensitivity(odczyt_HC_SR04, czas_impulsu)
##a,b=sensitivity(odczyt_VL53L1X, czas_impulsu)
print('a: ' + str(a) + 'b: ' + str(b))
#
a =0.00558503
b=0.00632165
float(b)
print(a)
x= np.linspace(100,1100,20)
y=a*x+b

plt.xlabel('distance[mm]')
plt.ylabel('pulse width[ms]')
plt.suptitle('Dopasowanie parametów')
plt.plot(x,y,'r-', label='y=(5.58e-03)*x + 6.32*e-03')
plt.plot(odczyt_HC_SR04, czas_impulsu,'bo')
plt.legend()
plt.show()

def resolution(d1):
    tmp = []
    for x in range(len(d1)-1):
        tmp.append(abs(d1[x+1]-d1[x]))
    res = min(tmp)
    print('Resolution: ' + str(res))

#resolution(odczyt_HC_SR04)
resolution(odczyt_VL53L1X)
def accuracy(odl, d1):
    accu = []
    for x in range (len(d1)):
        accu.append(abs(odl[x]-d1[x]))
    print('Accuracy: ' + str(max(accu)))

##odległość,czas_impulsu,_, odczyt_HC_SR04,_ = odczyt(dalmierz_pomiar_1)
##accuracy(odległość, odczyt_HC_SR04)
odległość,czas_impulsu,odczyt_VL53L1X, odczyt_HC_SR04,_ = odczyt(dalmierz_pomiar_1)
accuracy(odległość, odczyt_VL53L1X)

def sund_velo(odl, czas):
    v=[]
    for x in range (len(odl)):
        v.append(2*odl[x]/czas[x])
    avg = sum(v)/len(v)
    print('Predkosc dźwieku: ' + str(avg))

sund_velo(odczyt_VL53L1X,czas_impulsu)


print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n')

