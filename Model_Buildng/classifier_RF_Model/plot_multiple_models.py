import matplotlib.pyplot as plt
import numpy as np
import subprocess
from sys import argv
from collections import OrderedDict
acc = []
accMap = OrderedDict()
avg = []

#In order to run the test, we need to start as python random_forest_classifier.py
# Then, we concatenate the parsed_accounting (in form of csv) separated by space


for i in range (1,len(argv)):

    q = subprocess.Popen(['python', 'random_forest_classifier.py', argv[i]], stdout= subprocess.PIPE);
    out2 = q.communicate()
    thisAcc = [] # List of accuracy of the model on each month
    count = 0

    for j in argv:
        if not (j.endswith('.py') or (j == 'python')):
            p = subprocess.Popen(['python', 'rf_test.py', j], stdout=subprocess.PIPE);
            out, err = p.communicate()
            out_cond = p.stdout
            thisAcc.append(float(out.strip('\n')) * 100.0)

    acc.append(thisAcc)

    print (argv[i].split('_')[1])
    print ("Average accuracy of this model is ")
    print (np.mean(thisAcc))
    avg.append(np.mean(thisAcc))
    print ("\n")
    accMap[argv[i].split('_')[1]] = thisAcc

print ("List of accuracy of all models on different months")
print accMap

print ("Average of accuracy of all of the mdoels")
print (avg)

# ------------------------------------------------------------------
# Plot the average accuracies of all models (Model 1 - 8)

NUM_COLORS = 8

myFig = plt.figure()
ax = myFig.add_subplot(111)
cm = plt.get_cmap('gist_rainbow')
ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

for i in accMap.keys():
    plt.plot(accMap[i])

ind = np.arange(8)
plt.ylabel('Accuracy percentage(%)')
plt.xticks(ind, ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'June&Feb'])
plt.xlabel("Testing months")
plt.title("Prediction Accuracy from  different training models")
plt.legend(['Model 1(Jan dataset)', 'Model 2(Feb dataset)', 'Model 3(March dataset)',
            'Model 4(April dataset)', 'Model 5(May dataset)', 'Model 6(June dataset)',
            'Model 7(July dataset)', 'Model 8(Feb&June dataset)'])
plt.show()
