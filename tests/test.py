from svmloader import *

X, y = load_svmfile('satimage.scale', zero_based=False)
print(X.shape, X.nnz)