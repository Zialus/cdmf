import sys
import os

# to mtx format
fRatings="ratings.csv"
filePtr=open(fRatings,"r+")
file=open("ratings.mtx","w+")
print "start."


print "Getting the lines count."
count = len(open(fRatings, 'rU').readlines())
print "Lines count: %d" %count
ic = 0

arr1=[0 for x in range (0,27278)]
n=0

f = filePtr.readlines()
f.pop(0)	# skip the first line
for line in f:
                token=line.strip().split(',')
                uid = token[0]
                iid = token[1]
                score1 = token[2]
                score2 = token[3]

		if iid in arr1[0:n]:
			iid=arr1.index(iid)+1
		else:
			arr1[n]=iid
			iid=n+1
			n=n+1	

                file.write(str(uid)+"\t"+str(iid)+"\t"+str(score1)+"\n")
	        ic += 1
        	if (ic % 100000) == 0:
                	progress = float(ic)*100/float(count)
	                print "histogramming progress: %f%%" %progress
#                file.write("\t")
#                file.write(str(iid))
#                file.write("\t")
#                file.write(str(score1))
#                file.write("\n")
filePtr.close()
file.close()
print "end."

