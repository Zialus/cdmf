fRatings = "ratings.csv"
filePtr = open(fRatings, "r+")
file = open("ratings.mtx", "w+")

print "start."

print "Getting the lines count."
count = len(open(fRatings, 'rU').readlines())
print "Lines count: %d" % count

ic = 0
arr = [0 for x in range(0, 27278)]
n = 0

f = filePtr.readlines()
f.pop(0)  # skip the first line

for line in f:
    token = line.strip().split(',')
    uid = token[0]
    iid = token[1]
    score = token[2]

    if iid in arr[0:n]:
        iid = arr.index(iid)+1
    else:
        arr[n] = iid
        iid = n+1
        n = n+1

    file.write(str(uid))
    file.write("\t")
    file.write(str(iid))
    file.write("\t")
    file.write(str(score))
    file.write("\n")
    ic += 1
    if (ic % 100000) == 0:
        progress = float(ic)*100/float(count)
        print "histogramming progress: %f%%" % progress

filePtr.close()
file.close()
print "end."
