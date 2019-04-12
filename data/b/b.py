import random

print("start.")
dataName = "b.dat"
testFileName = "b.test"

file = open(dataName, "w+")
testFile = open(testFileName, "w+")

m = 1024000
n = m
uid = 0
iid = 0
score = 0
trainEntries = n+2*(m-1)
testEntries = trainEntries//100

meta = open("meta", "w+")
meta.write(str(m)+"\t"+str(n)+"\n")
meta.write(str(trainEntries)+"\t"+dataName+"\n")
meta.write(str(testEntries)+"\t"+testFileName+"\n")
meta.close()

print("Writing the train file...")
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
print("Done.")

file.seek(0)
head = [next(file) for x in range(testEntries)]
for line in head:
    testFile.write(line)

testFile.close()
file.close()
print("end.")
