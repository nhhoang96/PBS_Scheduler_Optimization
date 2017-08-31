import pandas as pd
import numpy as np
from sys import argv
import scipy.stats as stats
import matplotlib.pyplot as plt
stdev = []
max = []
min = []
estErrData = []
monthMap = {}
average = []

# The arguments include parsed accounting logs of different months

for i in range (1, len(argv)):
    print (argv[i])
    df = pd.read_csv(argv[i])
    monthName = argv[i].split('_')[1]

    temp = np.array(df['Estimation Error'].values)
    temp = temp/ 3600.0
    estErrData.append(df['Estimation Error'].values)
    monthMap[monthName] = temp
    max.append(temp.max())
    min.append(temp.min())
    stdev.append(temp.std())
    average.append(temp.mean())

labels = []

for i in range (1, len(argv)):
    monthName = argv[i].split('_')[1]
    labels.append(monthName)


# Draw bar graph
indices = np.array(range(len(stdev))) + 0.5
width = 0.5
plt.bar(indices, average, yerr =  stdev, width = width, color= 'r')
plt.xticks(indices + width/(len(stdev)), labels)
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.title('Monthly Statistical Analysis', fontsize = 20, fontweight = 'bold')
plt.xlabel('Months', color = 'b', fontsize = 14)
plt.ylabel('Estimation Error (hours)', color = 'b', fontsize = 14)
plt.savefig('Statistical Analysis Test.png')
plt.show()



