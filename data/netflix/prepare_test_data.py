
# coding: utf-8

# In[1]:

#prepare netflix data as an input to to cuMF
#data should be in ./data/netflix/
#assume input is given in text format
#each line is like 
#"user_id item_id rating"
import os
from six.moves import urllib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy import sparse


# In[2]:

# Step 1: Download the data.
url = 'http://www.select.cs.cmu.edu/code/graphlab/datasets/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

#from http://www.select.cs.cmu.edu/code/graphlab/datasets/
print "download netflix_mm and netflix_mme from the above URL first"
#train_data_file = 'netflix_mm.txt' #maybe_download('a.dat', 307198)
test_data_file = 'netflix_mme.txt' #maybe_download('a_t.dat', 0)

#netflix_mm and netflix_mme look like
'''
% Generated 25-Sep-2011
480189 17770 99072112
1 1  3
2 1  5
3 1  4
5 1  3
6 1  3
7 1  4
8 1  3

'''
#64000	150	1761439


m = 480189 
n = 17770
#nnz_train = 99072112
nnz_test = 1408395


# In[3]:

print "prepare test data"
#1-based to 0-based
test_i,test_j,test_rating = np.loadtxt(test_data_file,dtype=np.int32, skiprows=0, unpack=True)
R_test_coo = coo_matrix((test_rating,(test_i - 1,test_j - 1)))


# In[4]:

#for test data, we need COO format to calculate test RMSE
assert R_test_coo.nnz == nnz_test
R_test_coo.data.astype(np.float32).tofile('R_test_coo.data.bin')
R_test_coo.row.tofile('R_test_coo.row.bin')
R_test_coo.col.tofile('R_test_coo.col.bin')


