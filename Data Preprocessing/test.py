# import glob
# import pandas as pd
# import scipy.io
# import pywt
# import matplotlib.pyplot as plt



# cA, cD = pywt.dwt(b, 'db20', mode='periodic' )
# y = pywt.idwt(cA,cD, 'db20', mode = 'periodic')
# # print(cA)
# # print(cD)
# print(y)
# print(b)

# # coeffs = pywt.wavedec(b, 'db1', mode='periodic', level=2)
# # cA2, cD2, cD1 = coeffs
# # y = pywt.waverec(coeffs, 'db1', mode='periodic')

# # print(cA2)
# # print(cD2)
# # print(cD1)
# # print(y)
# # plt.figure(figsize=(12,6))
# # # plt.plot(y)
# # plt.plot(b)
# # plt.show()


import pywt
#from wavelets.wave_python.waveletFunctions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

mat = scipy.io.loadmat('E:\LAB\ScaleMind\Data_ECG\TrainingSet1\TrainingSet1\A0011.mat')
x = mat.get("ECG")
a = x[0][0]
# signal = a[2][6]
# # dataset = "https://raw.githubusercontent.com/taspinar/siml/master/datasets/sst_nino3.dat.txt"
# # df_nino = pd.read_table(dataset)
# # print(df_nino)

# N = len(signal)
# # time = np.arange(N/4500*10)
# t0=0
# dt=0.002
# time = np.arange(0, N) * dt + t0


def plot_wavelet(ax, time, signal, scales, waveletname = 'cmor', 
                 cmap = plt.cm.seismic):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    return yticks, ylim

scales = np.arange(1, 128)
# title = 'Wavelet Transform (Power Spectrum) of signal'
# ylabel = 'Period (years)'
# xlabel = 'Time'

# fig, ax = plt.subplots(figsize=(12, 6))
# plot_wavelet(ax, time, signal, scales)
# plt.show()

for i in range(11):
    signal = a[2][i]
    N = len(signal)
# time = np.arange(N/4500*10)
    dt=0.002
    time = np.arange(0, N) * dt
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_wavelet(ax, time, signal, scales)
    plt.show()
