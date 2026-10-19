import numpy as np
import pytest
import scipy.sparse as sp

import svmloader


DATA = "1 1:0.5 3:2\n0 2:1\n"


@pytest.fixture
def svm_file(tmp_path):
    path = tmp_path / "data.svm"
    path.write_text(DATA)
    return path


def test_load_svmfile_returns_matrix_then_labels(svm_file):
    X, y = svmloader.load_svmfile(str(svm_file), zero_based=False)
    assert isinstance(X, sp.csr_matrix)
    assert isinstance(y, np.ndarray)
    assert X.shape == (2, 3)
    assert X.nnz == 3
    assert y.tolist() == [1, 0]


def test_load_svmfiles_returns_interleaved_matrix_label_pairs(tmp_path):
    a = tmp_path / "a.svm"
    a.write_text("1 1:2\n")
    b = tmp_path / "b.svm"
    b.write_text("0 2:3\n")

    result = svmloader.load_svmfiles([str(a), str(b)], zero_based=False)

    assert len(result) == 4
    X0, y0, X1, y1 = result
    assert isinstance(X0, sp.csr_matrix)
    assert isinstance(X1, sp.csr_matrix)
    assert y0.tolist() == [1]
    assert y1.tolist() == [0]
    # nfeatures is the maximum index found on all the files
    assert X0.shape == X1.shape == (1, 2)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_save_svmfile_accepts_scipy_index_dtypes(tmp_path, index_dtype):
    X = sp.csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]]))
    X.indices = X.indices.astype(index_dtype)
    X.indptr = X.indptr.astype(index_dtype)
    path = tmp_path / "out.svm"

    svmloader.save_svmfile(str(path), X, np.array([1, 0]))

    assert path.read_text() == "1 2:1.5\n0 1:2\n"


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_save_load_roundtrip(tmp_path, index_dtype):
    X = sp.csr_matrix(np.array([[0.0, 1.5, 0.0], [2.0, 0.0, 3.0]]))
    X.indices = X.indices.astype(index_dtype)
    X.indptr = X.indptr.astype(index_dtype)
    y = np.array([1, 0])
    path = tmp_path / "roundtrip.svm"

    svmloader.save_svmfile(str(path), X, y)
    X2, y2 = svmloader.load_svmfile(str(path), zero_based=False)

    np.testing.assert_array_equal(y2, y)
    np.testing.assert_allclose(X2.toarray(), X.toarray())
