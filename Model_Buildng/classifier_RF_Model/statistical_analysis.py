import pandas as pd
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
stdev = []
max = []
min = []
monthMap = {}
average = []

# The arguments include the list of parsed accounting logs of different months

# Identify min, max, std, average from each month's data 
for i in range (1, len(argv)):
    print (argv[i])
    df = pd.read_csv(argv[i])
    monthName = argv[i].split('_')[1]
    estError = np.array(df['Estimation Error'].values)
    estError = estError/ 3600.0
    monthMap[monthName] = estError
    max.append(estError.max())
    min.append(estError.min())
    stdev.append(estError.std())
    average.append(estError.mean())

#-----------------------------------------------------------------------------
# Extract month name from "parsed accounting log files"
labels = []
for i in range (1, len(argv)):
    monthName = argv[i].split('_')[1]
    labels.append(monthName)

#----------------------------------------------------------------------------
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



