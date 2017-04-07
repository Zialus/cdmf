import sys
import os
from array import array

# count the number of items
fRatings="ratings.csv"
filePtr=open(fRatings,"r+")
iidArr = array('I')
n=0
print "Getting the lines count."
count = len(open(fRatings, 'rU').readlines())
print "Lines count: %d" %count
ic = 0
found=0

f = filePtr.readlines()
f.pop(0)        # skip the first line
for line in f:
	token=line.strip().split(',')
	uid = token[0]
	iid = token[1]
	score1 = token[2]
	score2 = token[3]
	try: 
		iidArr.index(int(iid))
		found = 1
	except ValueError: 
		found = 0
		iidArr.append(int(iid))
		n=n+1		
	ic += 1
	if (ic % 10000) == 0:
		progress = float(ic)*100/float(count)
		print "histogramming progress: %f%%" %progress

print "Lines count: %d" %n
filePtr.close()
