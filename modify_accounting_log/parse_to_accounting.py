import pandas as pd
from sys import argv
import os
import glob
import datetime

#The goal is to replace the Resource_List.wall_time from current accounting file
# with the newly predicted value from the model

# The argument has the following order:
#   1. Parsed CSV accounting file
#   2. Directory of input accounting files
#   3. Directory location of output accounting files

def read_into_txt():
    dest = argv[2]

    content = []
    statusArray = ["Q", "S", "B", "E"]
    files = glob.glob(dest + "/*")

    for f in files:
        with open(f, "r") as infile:
            parsedDate = f.split('/')
            print location
            output_File = open(argv[3] + parsedDate[-1], 'w')
            data = infile.readlines()

            for line in data:
                if ((line.split("/")[0].isdigit()) and (line.split(";")[1] == "E")):
                    element = line.split(' ')
                    sessionName = 0
                    qtime = 0
                    wallTimeLoc = 0
                    entity = element[1].split(';')[2].split('.')[0]

                    if not ('[' in entity):

                        for i in range(0,len(element)):
                            if ('session' in element[i]):
                                sessionName = int(element[i].split('=')[1])
                            elif ('qtime' in element[i]):
                                qtime = int(element[i].split('=')[1])
                            elif ('Resource_List.walltime' in element[i]):
                                wallTimeLoc =  i

                        if (((int(entity), qtime, sessionName) in mapModified.keys())):
                            element[wallTimeLoc] = 'Resource_List.walltime=' + str(mapModified[(int(entity), qtime, sessionName)])

                    writeLine = ' '.join(element)
                    output_File.write(writeLine)

                else:
                    output_File.write(line)

fields = ['Entity', 'qtime', 'session', 'Modified Resource_List.walltime']
df = pd.read_csv(argv[1], usecols = fields)
mapModified = {}
entityArr = df['Entity'].values
qtimeArr = df['qtime'].values
sessionArr = df['session'].values
wallTimeArr = df['Modified Resource_List.walltime'].values

for i in range (len(wallTimeArr)):
    correctedTime = datetime.timedelta(0, wallTimeArr[i])
    mapModified[entityArr[i], qtimeArr[i], sessionArr[i]] = correctedTime

read_into_txt()
