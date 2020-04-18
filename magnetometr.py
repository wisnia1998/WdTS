import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataX = pd.read_csv('magneto_data_X.csv', header = 0)##X_data I Hx Hy Hz
dataY = pd.read_csv('magneto_data_Yc.csv', header = 0)##X_data I Hx Hy Hz
dataZ = pd.read_csv('magneto_data_Zc.csv', header = 0)##X_data I Hx Hy Hz
dHist = pd.read_csv('magneto_data_histereza.csv', header = 0)##magneto_data_histereza I Hx Hy Hz 
ch1 = pd.read_csv('charakterystyka1.csv', header = 0)
ch2 = pd.read_csv('charakterystyka2.csv', header = 0)
ch3 = pd.read_csv('charakterystyka3.csv', header = 0)
ch4 = pd.read_csv('charakterystyka4.csv', header = 0)

#1st
def rysuj(data):
    fig, axs = plt.subplots(3)
    axs[0].plot(data['I']*31.9,data['Hx']/230,'r')
    axs[0].plot(data['I']*31.9,data['Hx']/230,'b.')
    axs[1].plot(data['I']*31.9,data['Hy']/230,'r')
    axs[1].plot(data['I']*31.9,data['Hy']/230,'g.')
    axs[2].plot(data['I']*31.9,data['Hz']/230,'r')
    axs[2].plot(data['I']*31.9,data['Hz']/230,'b.')
    axs[0].set_title('X scale')
    axs[1].set_title('Y scale')
    axs[2].set_title('Z scale')
    plt.ylabel('H_measured')
    plt.xlabel('H_set')

def rys3(data1,data2,data3):
    fig, axs = plt.subplots(3);
    axs[0].plot(data1['I']*31.9,data1['Hx']/230,'r')
    axs[0].plot(data1['I']*31.9,data1['Hx']/230,'b.')
    axs[1].plot(data2['I']*31.9,data2['Hy']/230,'r')
    axs[1].plot(data2['I']*31.9,data2['Hy']/230,'g.')
    axs[2].plot(data3['I']*31.9,data3['Hz']/230,'r')
    axs[2].plot(data3['I']*31.9,data3['Hz']/230,'k.')
    axs[2]
    axs[0].set_title('X scale')
    axs[1].set_title('Y scale')
    axs[2].set_title('Z scale')
    plt.ylabel('H_measured[G]')
    plt.xlabel('H_set[G]')
#1st end

#2nd
def func(x, a, b):
    return a*x+b

def f(xr,y,t):
    x = [i for i in xr if i<=8.1]
    y = np.array(y[:len(x)])
    x = np.array(x)
    param, _ = curve_fit(func, x, y)
    a = param[0]
    b = param[1]
    plt.figure()
    plt.plot(x,y,'ro')
    plt.plot(x, func(x, a, b))
    plt.xlabel('Wartosc zadana [G]')
    plt.ylabel('Wartosc zmierzona [G]')
    plt.title(t)
    plt.text(1, 2.2, r'a=' + str(a))
    plt.text(1, 2, r'b=' + str(b))
    print(a)
    print(b)
#2nd end

#3rd cross-talk
def cross_talk(dI,dP,d1,d2):
    I = [ i for i in dI if i <= 8.1]
    P = dP[:len(I)]
    d1 = d1[:len(I)]
    d2 = d2[:len(I)]
    cY=[]
    cZ=[]
    for i in range (len(P)-1):
        cY.append((abs((d1[i+1]-d1[i]))/abs(P[i]))/16*100)
        cZ.append((abs((d2[i+1]-d2[i]))/abs(P[i+1]))/16*100)
    plt.figure()
    plt.title('Y-axis cross-talk')
    plt.ylabel('[%]')
    plt.xlabel('Field [G]')
    plt.plot(I[:-1],cY[:],'.')
    plt.text(4.5, 0.47,'cross-talk: ' + str(round(max(cY[:]),2))+'%')
    plt.figure()
    plt.title('Z-axis cross-talk')
    plt.ylabel('[%]')
    plt.xlabel('Field [G]')
    plt.plot(I[:-1],cZ[:],'.')
    plt.text(4, 0.48,'cross-talk: ' + str(round(max(cZ[:]),2))+'%')
    print(cY)
#3rd end

#4th
def hist(dI,dP,d1,d2):
    I = [ i for i in dI if i <= 8.1]
    p = dP[:len(I)]
    d1 = d1[:len(I)]
    d2 = d2[:len(I)]
    fit_params, covariance_matrix = curve_fit(func, I, p)
    f = func(np.array(I), *fit_params)
    plt.title("Histereza")
    plt.xlabel("Prąd[A]")
    plt.ylabel("Pomiar [LSB]")
    plt.text(-0.8 ,7.5 , 'Histereza = ' + str(max(abs(f-p))))
    plt.show()
#4th end
    
#5th
##fit function
def sin(x,a,b,c,d):
    return a * np.sin(b * x * np.pi / 180 + c) + d
def cos ( x , a , b , c , d ):
    return a * np.cos(b * x * np.pi / 180 + c) + d
def tanh ( x , a , b , c , d ):
    return a * np.tanh(b * x * np.pi / 180 + c) + d
def gauss(x, a, b, c,d):
       return a * np.exp(-(x - b)**2.0 / (2 * c**2))+d


def z5(fs,d1,d2):
    x,y=[],[]
    for A in d1:
        x.append(float(A))
    for R in d2:
        y.append(float(R))
    x=np.array(x)
    y=np.array(y)
    fit_params, covariance_matrix = curve_fit(fs, x, y,p0 =[ 50 , 1 , 1 , 500 ]) 
    plt.figure()
    plt.title( "Charakterystyka 3" )
    plt.xlabel( "Kąt[stopnie]" )
    plt.ylabel( "Rezystancja[Ohm]" )
    plt.plot(x,y, 'ro')
    plt.plot(x, fs(x, *fit_params), 'b')
    plt.text(- 182.5 , 45 , ( 'Funkcja dopasowania = ' + str ( round (fit_params[ 0 ], 2 ))
    + '*exp(-(x - ' + str ( round (fit_params[ 1 ], 2 )) + ')**2.0 / (2 * '
    + str ( round (fit_params[ 2 ], 2 )) + '**2))' + str ( round (fit_params[ 3 ], 2 ))))
##    plt.text(- 182.5 , 45 , ( 'Funkcja dopasowania = ' + str ( round (fit_params[ 0 ], 2 )) + '*tanh('
##    + str ( round (fit_params[ 1 ], 2 )) + '*x*pi/180+' + str ( round (fit_params[ 2 ], 2 )) + ')+' +
##    str ( round (fit_params[ 3 ], 2 ))))
#5th end
   
#z5(sin,ch1['Angle'],ch1['Resistance'])
#z5(cos,ch2['Angle'],ch2['Resistance'])
z5(gauss,ch3['Angle'],ch3['Resistance'])
#z5(tanh,ch4['H'],ch4['Resistance'])

#hist(dHist['I'],dHist['Hx']/230,dHist['Hy'],dHist['Hz'])

#cross_talk(dataX['I']*31.9,dataX['Hx']/230,dataX['Hy']/230,dataX['Hz']/230)
#cross_talk(dataY['I']*31.9,dataY['Hy']/230,dataY['Hx']/230,dataY['Hz']/230)
#cross_talk(dataZ['I']*31.9,dataZ['Hz']/230,dataZ['Hy']/230,dataZ['Hx']/230)

#f(dataX['I']*31.9,dataX['Hx']/230,'Oś X')
#f(dataY['I']*31.9,dataY['Hx']/230,'Oś Y')
#f(dataZ['I']*31.9,dataZ['Hx']/230,'Oś Z')

#rysuj(dataX)
#rysuj(dataY)
#rysuj(dataZ)
#rys3(dataX,dataY,dataZ)
plt.show()
