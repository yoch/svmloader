from svmloader import *

X, y = load_svmfile('satimage.scale', zero_based=False)
print(X.shape, X.nnz)

X, y = load_svmfile('yeast_train.svm', zero_based=False, multilabels=True)
print(X.shape, X.nnz)