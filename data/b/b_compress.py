import numpy as np
from scipy import sparse




m = 1024000
n = 1024000
nnz_train = 3071998
nnz_test = 30719

train_data_file = "b.dat"
test_data_file = "b.test"




print("preparing test data")
test_user, test_item, test_rating = np.loadtxt(test_data_file, delimiter='\t',
                                               dtype=[('f0', np.int32), ('f1', np.int32), ('f2', np.float)],
                                               skiprows=0, unpack=True)

print("preparing training data")
train_user, train_item, train_rating = np.loadtxt(train_data_file, delimiter='\t',
                                                  dtype=[('f0', np.int32), ('f1', np.int32), ('f2', np.float)],
                                                  skiprows=0, unpack=True)




print(test_user)
print(test_item)
print(test_rating)
print("")
print(np.max(test_user))
print(np.max(test_item))
print(np.max(test_rating))
print("")
print(np.min(test_user))
print(np.min(test_item))
print(np.min(test_rating))
print("")
print(np.unique(test_user).size)
print(np.unique(test_item).size)
print(np.unique(test_rating).size)
print("")
print(test_user.size)

assert test_user.size == nnz_test




print(train_user)
print(train_item)
print(train_rating)
print("")
print(np.max(train_user))
print(np.max(train_item))
print(np.max(train_rating))
print("")
print(np.min(train_user))
print(np.min(train_item))
print(np.min(train_rating))
print("")
print(np.unique(train_user).size)
print(np.unique(train_item).size)
print(np.unique(train_rating).size)
print("")
print(train_user.size)

assert train_user.size == nnz_train
assert np.max(train_user) == m
assert np.max(train_item) == n




#1-based to 0-based
R_test_coo = sparse.coo_matrix((test_rating, (test_user - 1, test_item - 1)))
assert R_test_coo.nnz == nnz_test

outfile_test = open("test.txt", 'w')
for i in range(nnz_test):
    outfile_test.write(str(test_user[i]) + " " + str(test_item[i]) + " " + str(test_rating[i]) + "\n")
outfile_test.close()



# for test data, we need COO format to calculate test RMSE
R_test_coo.data.astype(np.float32).tofile('R_test_coo.data.bin')
R_test_coo.row.tofile('R_test_coo.row.bin')
R_test_coo.col.tofile('R_test_coo.col.bin')

test_data = np.fromfile('R_test_coo.data.bin', dtype=np.float32)
test_row = np.fromfile('R_test_coo.row.bin', dtype=np.int32)
test_col = np.fromfile('R_test_coo.col.bin', dtype=np.int32)




print(R_test_coo.data)
print(R_test_coo.row)
print(R_test_coo.col)
print("")
print(test_data)
print(test_row)
print(test_col)




print(np.max(R_test_coo.data))
print(np.max(R_test_coo.row))
print(np.max(R_test_coo.col))
print("")
print(np.min(R_test_coo.data))
print(np.min(R_test_coo.row))
print(np.min(R_test_coo.col))
print("")
print(np.unique(test_user).size)
print(np.unique(R_test_coo.row).size)
print(np.unique(test_item).size)
print(np.unique(R_test_coo.col).size)




#1-based to 0-based
R_train_coo = sparse.coo_matrix((train_rating, (train_user - 1, train_item - 1)))
assert R_train_coo.nnz == nnz_train

outfile_train = open("train.txt", 'w')
for i in range(nnz_train):
    outfile_train.write(str(train_user[i]) + " " + str(train_item[i]) + " " + str(train_rating[i]) + "\n")
outfile_train.close()



# for training data, we need COO format to calculate training RMSE
# we need CSR format R when calculate X from \Theta
# we need CSC format of R when calculating \Theta from X
R_train_coo.data.astype(np.float32).tofile('R_train_coo.data.bin')
R_train_coo.row.tofile('R_train_coo.row.bin')
R_train_coo.col.tofile('R_train_coo.col.bin')

R_train_csr = R_train_coo.tocsr()
R_train_csc = R_train_coo.tocsc()

R_train_csr.data.astype(np.float32).tofile('R_train_csr.data.bin')
R_train_csr.indices.tofile('R_train_csr.indices.bin')
R_train_csr.indptr.tofile('R_train_csr.indptr.bin')
R_train_csc.data.astype(np.float32).tofile('R_train_csc.data.bin')
R_train_csc.indices.tofile('R_train_csc.indices.bin')
R_train_csc.indptr.tofile('R_train_csc.indptr.bin')




train_data = np.fromfile('R_train_coo.data.bin', dtype=np.float32)
train_row = np.fromfile('R_train_coo.row.bin', dtype=np.int32)
train_col = np.fromfile('R_train_coo.col.bin', dtype=np.int32)

train_csc_data = np.fromfile('R_train_csc.data.bin', dtype=np.float32)
train_csc_indices = np.fromfile('R_train_csc.indices.bin', dtype=np.int32)
train_csc_indptr = np.fromfile('R_train_csc.indptr.bin', dtype=np.int32)

train_csr_data = np.fromfile('R_train_csr.data.bin', dtype=np.float32)
train_csr_indices = np.fromfile('R_train_csr.indices.bin', dtype=np.int32)
train_csr_indptr = np.fromfile('R_train_csr.indptr.bin', dtype=np.int32)




print(R_train_coo.data)
print(R_train_coo.row)
print(R_train_coo.col)
print("")
print(train_data)
print(train_row)
print(train_col)
print("")
print(R_train_csr.data)
print(R_train_csr.indices)
print(R_train_csr.indptr)
print("")
print(train_csr_data)
print(train_csr_indices)
print(train_csr_indptr)
print("")
print(R_train_csc.data)
print(R_train_csc.indices)
print(R_train_csc.indptr)
print("")
print(train_csc_data)
print(train_csc_indices)
print(train_csc_indptr)




print(np.max(R_train_coo.data))
print(np.max(R_train_coo.row))
print(np.max(R_train_coo.col))
print("")
print(np.min(R_train_coo.data))
print(np.min(R_train_coo.row))
print(np.min(R_train_coo.col))
print("")
print(np.unique(train_user).size)
print(np.unique(R_train_coo.row).size)
print(np.unique(train_item).size)
print(np.unique(R_train_coo.col).size)




print("write extra meta file")
outfile_meta = open("meta_modified_all", 'w')
outfile_meta.write(str(m) + " " + str(n) + "\n" + str(nnz_train) + "\n")
outfile_meta.write("""R_train_coo.data.bin
R_train_coo.row.bin
R_train_coo.col.bin
R_train_csr.indptr.bin
R_train_csr.indices.bin
R_train_csr.data.bin
R_train_csc.indptr.bin
R_train_csc.indices.bin
R_train_csc.data.bin
""")
outfile_meta.write(str(nnz_test) + "\n")
outfile_meta.write("""R_test_coo.data.bin
R_test_coo.row.bin
R_test_coo.col.bin
""")
outfile_meta.close()




print("writing extra meta file")
outfile_meta = open("meta", 'w')
outfile_meta.write(str(m) + " " + str(n) + "\n")
outfile_meta.write(str(nnz_train) + " " + "train.txt\n")
outfile_meta.write(str(nnz_test) + " " + "test.txt\n")
outfile_meta.close()
