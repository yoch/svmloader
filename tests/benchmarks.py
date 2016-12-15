def load(loader, filename):
    X, y = loader(filename)
    print('\t', X.shape, X.nnz)


if __name__ == '__main__':
    import sys
    from timeit import timeit
    from time import sleep
    from functools import partial
    import os.path
    import numpy
    import scipy.sparse
    from svmloader import load_svmfile, load_svmfiles
    from sklearn.datasets import load_svmlight_file, load_svmlight_files

    for filename in sys.argv[1:]:
        print(os.path.basename(filename))
        for loader in [load_svmfile, load_svmlight_file]:
            sleep(5)
            loader_ = partial(loader, zero_based=False)
            t = timeit(lambda: load(loader_, filename), number=1)
            print('\t', loader.__name__, ':', t, 's')
