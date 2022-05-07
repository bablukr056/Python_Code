import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
#-------------------------------------------file_opening------------
f1 = pd.read_csv("cvd_dataset.csv",sep=",")
#--------------------visualizing of cases occurs according to the age----------¶
rcParams['figure.figsize'] = 11, 8
print(sns.countplot(x='age', hue='cardio', data = f1, palette="Set2"))
