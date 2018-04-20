import os
import sys
import random

dataName="c.dat"
#dataName="c.mtx"
file=open(dataName,"w+")
m=1024000
n=m
uid=0
iid=0
score=0
print "start."

print "Writing the meta file..."
meta=open("meta", "w+")
meta.write(str(m)+"\t"+str(n)+"\n")
meta.write(str(2*m)+"\t"+dataName+"\n") 
meta.close();
print "Done."

#file.write("%%MatrixMarket matrix coordinate real general\n")
#head = str(m) + "\t" + str(n) + "\t" + str(2*m) + "\n"
#file.write(head)

for x in range(m): # generate two entries for each row
	uid = x+1
	iid = 1
	score = random.randint(1,5)
	file.write(str(uid))
	file.write("\t")
	file.write(str(iid))
	file.write("\t")
	file.write(str(score))
	file.write("\n")
	#uid = x+1
	iid = random.randint(2,n)
	score = random.randint(1,5)
	file.write(str(uid))
	file.write("\t")
	file.write(str(iid))
	file.write("\t")
	file.write(str(score))
	file.write("\n")

file.close()
print "end."
