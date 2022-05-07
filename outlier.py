import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics


ouliers=[]
def detect_outlier(data):
    threshold=3
    mean=np.mean(data)
    std=np.std(data)

    for i in data:
        z_scor=(i-mean)/std
        if np.abs(z_scor)>threshold:
            outlier.append(i)

    return outlier

f1 = open("cardio_train.csv","r")
while True:
    line  = f1.readline()[0:]
    if not line:
        break
    line= re.split(';', line)
    data=round(int(line[1])/365)
    mean=np.mean(data)
    print(mean)

