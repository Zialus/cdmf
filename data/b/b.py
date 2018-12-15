import random

dataName = "b.dat"
# dataName = "b.mtx"
file = open(dataName, "w+")
m = 1024000
n = m
uid = 0
iid = 0
score = 0
print("start.")

print("Writing the meta file...")
meta = open("meta", "w+")
meta.write(str(m)+"\t"+str(n)+"\n")
meta.write(str(n+2*(m-1))+"\t"+dataName+"\n")
meta.close()
print("Done.")

# file.write("%%MatrixMarket matrix coordinate real general\n")
# head = str(m) + "\t" + str(n) + "\t" + str(n+2*(m-1)) + "\n"
# file.write(head)

for x in range(1, m+1):
    if x == m:
        for y in range(1, n+1):
            uid = x
            iid = y
            score = random.randint(1, 5)
            file.write(str(uid))
            file.write("\t")
            file.write(str(iid))
            file.write("\t")
            file.write(str(score))
            file.write("\n")
    else:  # generate two entries for each row
        uid = x
        iid = random.randint(1, m-1)
        score = random.randint(1, 5)
        file.write(str(uid))
        file.write("\t")
        file.write(str(iid))
        file.write("\t")
        file.write(str(score))
        file.write("\n")
        uid = x
        iid = n
        score = random.randint(1, 5)
        file.write(str(uid))
        file.write("\t")
        file.write(str(iid))
        file.write("\t")
        file.write(str(score))
        file.write("\n")

file.close()
print("end.")
