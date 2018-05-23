filePtr = open("training.ratings", "r+")
file = open("traindata.txt", "w+")

arr1 = [0 for x in range(0, 480190)]
n = 0

print "start."
for line in filePtr:
        token = line.strip().split()
        uid = token[0]
        iid = token[1]
        score = token[2]
        if uid in arr1[0:n]:
            uid = arr1.index(uid)+1
        else:
            arr1[n] = uid
            uid = n+1
            n = n+1
        file.write(str(uid))
        file.write("\t")
        file.write(str(iid))
        file.write("\t")
        file.write(str(score))
        file.write("\n")
print "end."

filePtr.close()
file.close()
