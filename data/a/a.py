import random

dataName = "a.dat"
# dataName = "a.mtx"
file = open(dataName, "w+")
m = 64000
n = 150
uid = 0
iid = 0
score = 0
print "start."

print "Writing the meta file..."
meta = open("meta", "w+")
meta.write(str(m)+"\t"+str(n)+"\n")
meta.write(str(n+2*(m-1))+"\t"+dataName+"\n")
meta.close()
print "Done."

# file.write("%%MatrixMarket matrix coordinate real general\n")
# head = str(m) + "\t" + str(n) + "\t" + str(n+2*(m-1)) + "\n"
# file.write(head)

for x in range(m):
    if x+1 == 1:
        for y in range(n):
            uid = x+1
            iid = y+1
            score = random.randint(1, 5)
            file.write(str(uid))
            file.write("\t")
            file.write(str(iid))
            file.write("\t")
            file.write(str(score))
            file.write("\n")
    else:
        uid = x+1
        iid = 1
        score = random.randint(1, 5)
        file.write(str(uid))
        file.write("\t")
        file.write(str(iid))
        file.write("\t")
        file.write(str(score))
        file.write("\n")
        uid = x+1
        iid = random.randint(2, n)
        score = random.randint(1, 5)
        file.write(str(uid))
        file.write("\t")
        file.write(str(iid))
        file.write("\t")
        file.write(str(score))
        file.write("\n")

file.close()
print "end."
