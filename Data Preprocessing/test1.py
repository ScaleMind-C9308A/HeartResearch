import glob
import pandas as pd
import scipy.io
import pywt
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('E:\LAB\ScaleMind\Data_ECG\TrainingSet1\TrainingSet1\A0011.mat')
x = mat.get("ECG")
a = x[0][0]
b = a[2][6]

# cA, cD = pywt.dwt(b, 'db20', mode='periodic' )
# y = pywt.idwt(cA,cD, 'db20', mode = 'periodic')
# print(cA)
# print(cD)
# print(y)
# print(b)

coeffs = pywt.wavedec(b, 'db4s', mode='periodic', level=2)
cA2, cD2, cD1 = coeffs
y = pywt.waverec(coeffs, 'db4', mode='periodic')

# print(cA2)
# print(cD2)
# print(cD1)
# print(y)
plt.figure(figsize=(12,6))
# plt.plot(y)
plt.plot(cA2)
plt.plot(cD1)
plt.plot(cD2)
plt.show()