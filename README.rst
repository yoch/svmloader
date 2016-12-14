**svmloader** is a simplist but very fast python module (written in
cython) to load sparse data written at libsvm format.

It is not functionnaly equivalent to
"sklearn.datasets.load_svmlight_file", and handle only the simplest
cases. *labels* are supposed to be of integer type, and data is parsed
as *numpy.float64* type.


svmloader.load_svmfile(filename, nfeatures=None, zero_based=True)

   Load a sparse matrix from filename at svmlib format.

   Parameters:
      * **filename** (*str*) -- the file name

      * **nfeatures** (*int*) -- the number of columns (infered from
        file if is None)

      * **zero_based** (*bool*) -- indicates if columns indexes are
        zero-based or one-based

   Returns:
      (labels, sparse_matrix) tuple

   Return type:
      ("numpy.ndarray", "scipy.sparse.csr_matrix")

svmloader.load_svmfiles(filenames, zero_based=True)

   Load a sparse matrix list from list of filenames at svmlib format.
   The number of features will be infered from the maximum indice
   found on all files.

   Parameters:
      * **filenames** (*list*) -- the list of files names

      * **zero_based** (*bool*) -- indicates if columns indexes are
        zero-based or one-based

   Returns:
      a list [labels_0, matrix_0, .., labels_n, matrix_n]
