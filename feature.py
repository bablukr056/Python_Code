import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
#-----------------------------------------------------------------
df = pd.read_csv('combined.csv', sep=",")

plt.rcParams['figure.figsize'] = (30, 15)
sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap="YlGnBu")
plt.title('Corelation Between Features', fontsize = 30)
plt.show()
