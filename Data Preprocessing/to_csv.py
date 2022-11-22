import glob
import scipy.io as si
import pandas as pd

data = 'E:\LAB\ScaleMind\Data_ECG\TrainingSet1\TrainingSet1\A0011.mat'
mat = si.loadmat(data)
x = mat.get("ECG")
a = x[0][0][2]
for i in range(2):
    df = pd.DataFrame(a[i])
    df.to_csv(f'E:\git\HeartResearch\Data Preprocessing\\{i}.csv')
