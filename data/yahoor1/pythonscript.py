import sys
import os

filePtr=open("ydata-ymusic-user-artist-ratings-v1_0.txt","r+")
file=open("ydatamusic.txt","w+")
#arr1=[0,0,0,0,0]
arr1=[0 for x in range (0,99000)]
n=0
print "start."
for line in filePtr:
		token=line.strip().split()
		uid = token[0]
		iid = token[1]
		score = token[2]
		if int(score)!=255:
			if iid in arr1[0:n]:
				iid=arr1.index(iid)+1
			else:
				arr1[n]=iid
				iid=n+1
				n=n+1				
			file.write(str(uid))
			file.write("\t")
			file.write(str(iid))
			file.write("\t")
			file.write(str(score))
			file.write("\n")
filePtr.close()
file.close()
print "end."
