import os.path
from svmloader import *

DIR = os.path.dirname(__file__)

X, y = load_svmfile(os.path.join(DIR, 'satimage.scale'), zero_based=False)
print(X.shape, X.nnz)

X, y = load_svmfile(os.path.join(DIR, 'yeast_train.svm'), zero_based=False, multilabels=True)
print(X.shape, X.nnz)
