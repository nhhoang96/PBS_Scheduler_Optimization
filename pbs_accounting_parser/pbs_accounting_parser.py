from sys import argv,exit
from string import split,join
import csv
import os
import time
import calendar
from datetime import timedelta
import pandas as pd
from itertools import cycle, islice
import numpy as np
import glob
import collections

title = []
extra_features = {}


# Argument inputs are as follows:
# 	1. Directory of accounting files
#	2. Name of text output file
#	3. Name of output csv file

# Read all accounting logs into one single text file
def read_into_txt(dest):
		content = []
		statusArray = ["Q", "S", "B", "E"];
		read_files = glob.glob(dest + "/*")

		with open(argv[2], "wb") as outfile:
			for f in read_files:
				with open(f, "r") as infile:
					data = infile.readlines()
					for line in data:
						if ((line.split("/")[0].isdigit()) and (line.split(";")[1] == "E") ):
							content.append(line)
							outfile.write(line)
							outfile.write(infile.read())

# Parse messages of accounting logs
def parse_acct_record(m):
	squote = 0
	dquote = 0
	paren = 0
	key = ""
	value = ""
	in_key = 1
	rval = {}

	for i in range(0, len(m)):
		#safety checks
		if in_key < 0:
			raise Exception("Unexpected Happened")
		if in_key < 1 and key == "":
			raise Exception("Null Key")

		#parens seem to be super-quotes
		if m[i] == '(':
			paren = paren + 1
		if m[i] == ')':
			paren = paren - 1
		#single quotes are the next strongest escape character
		if m[i] == '\'':
			if squote > 0:
				squote = squote - 1
			else:
				if dquote > 1:
					raise Exception("Don't think this can happen")
				squote = squote + 1
		#then double quotes
		if m[i] == '"':
			if dquote > 0 and squote == 0:
				dquote = dquote - 1
			else:
				dquote = dquote + 1
		#last, equal signs
		if m[i] == '=' and squote == 0 and dquote == 0 and paren == 0:
			if value is "":
				in_key = 0
				continue
			else:
				if not (m[i] == '=' and in_key == 0):
					#pretty sure you can't have an equal in a key
					#print m
					raise Exception("Unhandled Input", m[i])
		if m[i] == ' ' and (squote > 0 or dquote > 0 or paren > 0):
			if in_key == 1:
				key += m[i]
				continue
			else:
				value += m[i]
				continue
		if m[i] == ' ' and in_key==0:
			#print "Key: " + key
			#print "Value: " + value
			if not key in rval:
				rval[key] = value
			#else:
				#aise Exception("Duplicate Key")
				#print b

			in_key = 1
			key = ""
			value = ""
			continue
		if m[i] == ' ':
			continue
			raise Exception("Unexpected whitespace")
		if in_key == 1:
			key += m[i]
		if in_key == 0:
			value += m[i]
	if in_key == 1 and len(key) > 1:
		#raise Exception("Partial Record Detected", argv[1])
		print "Warning: Gibberish: " + key
	rval[key.rstrip('\n')] = value.rstrip('\n')
	return rval

# Parse "SELECT" field of accounting logs
def parseSelect(v):
	element = v.split(":")
	numCores = element[0]print (feature_array)

	new_feature = {}
	for e in range (1, len(element), 1):
		comp = element[e].split("=")
		if ((comp[0] != 'mpiprocs') and (comp[0] != 'ncpus')):
			new_feature[comp[0]] = comp[1]
	return numCores, new_feature

def main():
	do_output = 0
	if len(argv) < 3:
		print "Missing argument"
		exit(1)
	if len(argv) > 3:
		do_output = 1
		read_into_txt(argv[1])
		outputFile = open('temp.csv', 'w')
		output_writer = csv.writer(outputFile)
		check = 1 # The title has not been written yet
		accounting_file = open(argv[2], 'r')
		record_id = -1

		#Per record
		for entry in accounting_file:
			record_id = record_id + 1
			fields = split(entry, ';')

			# Time that records are written
			rtime = time.strptime(fields[0], "%m/%d/%Y %H:%M:%S") #localtime() -- local time
			rtime = calendar.timegm(rtime)
			etype = fields[1] #[LQSED]
			job_num = fields[2].split('.') #"license" or job number
			entity = job_num[0]
			message = join(fields[3:])

			if etype == 'L':
				continue #PBS license stats? not associated with a job
			rec = parse_acct_record(message)

			if ((etype == 'E') and ( not ('[]' in entity))):
				if do_output == 1:
					diff = 0

					# Ignore jobs with Resource_list.inception field
					if ('Resource_List.inception' in rec.keys()):
							pass

					# Ignore jobs with specified Resource_List.mem since not all users specify this information
					elif ('Resource_List.mem' in rec.keys()):
							pass
							#Ignore
					else:
						if ((len(rec.keys()) != 28) and ('Resource_List.mpiprocs' in rec.keys()) and ('account' in rec.keys())):
							pass
							# Ignore

						# Submitted jobs without specifying account name, ignore since it is old data
						elif not ('account' in rec.keys()):
							pass

						else:
							info = []
							info.append(record_id)
							info.append(rtime)
							info.append(entity)

							if (cprint (feature_array)
heck == 1):
								title.append("ID")
								title.append("rtime")
								title.append("Entity")

							qtime = 0
							eligible = 0
							waitTime = 0

							if 'Resource_List.walltime' in rec.keys():
								#Assing mpiprocs = 0 if it is not specified by users
								if not ('Resource_List.mpiprocs' in rec):
									rec['Resource_List.mpiprocs'] = 0

								new_features = {}
								print (feature_array)
for k,v in rec.iteritems():

									if ((check == 1) and (k != 'Resource_List.select')):
										title.append(k)

									if ((k == 'resources_used.walltime') or (k == 'Resource_List.walltime')
											or (k == 'resources_used.cput')):
										timeElements = v.split(":")
										corTime = timedelta(hours=int(timeElements[0]), minutes=int(timeElements[1]),
															seconds=int(timeElements[2])).total_seconds()

										if (k == 'resources_used.walltime'):
											diff = diff - corTime
										elif (k == 'Resource_List.walltime'):
											diff = diff + corTime
										info.append(corTime)

									elif (k == 'resources_used.mem' or k == 'resources_used.vmem'):
										#Remove kb to make the filed numeric
										if ('kb' in v):
											v = v.replace('kb', '')
											info.append(v)

									elif (k == 'Resource_List.select'):
										numCores,new_features = parseSelect(v)
										info.append(numCores)

										if (check == 1):
											title.append("Number of nodes")

									elif (k == 'qtime'):
										info.append(v)
										qtime = v

									elif (k == 'etime'):
										eligible  = v
										info.append(v)

										if ((long(eligible) - long(qtime)) > 0):
											waitTime = long(eligible) - long(qtime)

									else:
										info.append(v)

								info.append(waitTime)
								info.append(diff)

								if (check == 1):
									title.append("Waiting Time")
									title.append("Estimation Error")

								if (len(new_features) != 0):
									#extra_features[record_id] = []
									temp = []
									new_features = collections.OrderedDict(sorted(new_features.items(), reverse=True))
									print (new_features)
									for k, v in new_features.iteritems():
										temp.append([k, v])
										extra_features[record_id] = temp

										if not (k in title):
											title.append(k)

							if (check == 1):
								output_writer.writerow(title)
								check = 0
							output_writer.writerow(info)

				if do_output == 0:
					pass

		outputFile = open(argv[3], 'w')
		output_writer = csv.writer(outputFile)

		inputFile = open('temp.csv', 'r')
		reader = csv.reader(inputFile)
		next(reader)
		output_writer.writerow(title)

		for line in reader:

			if (int(line[0]) in extra_features.keys()):

				row_key = int(line[0])
				for i in range(len(extra_features[row_key])):
					index = title.index(extra_features[row_key][i][0])
					if (index > len(line)):
						for j in range (0, index - len(line),1):
							line.insert(len(line)+ j, '0')

					line.insert(index, extra_features[row_key][i][1])
				output_writer.writerow(line)

if __name__ == "__main__":
	main()
