import pandas as pd
import matplotlib.pyplot as plt

# Import and Process Data
data = pd.read_csv("data/winequality-white.csv",sep=';')

# Histogram
for x in range(0,len(data.columns)):
    plt.subplot(4,4,x+1)
    plt.hist(data[data.columns[x]],label=data.columns[x])
    plt.legend(loc='upper_right')
    plt.show()

# Pairwise scatterplot
axes = pd.tools.plotting.scatter_matrix(data[data.columns[0:4]], alpha=0.2)
axes = pd.tools.plotting.scatter_matrix(data[data.columns[4:8]], alpha=0.2)
axes = pd.tools.plotting.scatter_matrix(data[data.columns[8:12]], alpha=0.2)

# Scatterplot: Quality vs. all other features
for x in range(0,len(data.columns)-1):
    plt.subplot(4,3,x+1)
    plt.scatter(data[data.columns[0]],data['quality'],alpha=0.2,label=data.columns[x])
    plt.legend(loc='upper_right')
    plt.show()
