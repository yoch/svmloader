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


def test_docstrings_document_matrix_first():
    assert "(X, y)" in svmloader.load_svmfile.__doc__
    assert "[matrix_0, labels_0" in svmloader.load_svmfiles.__doc__


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_save_svmfile_accepts_read_only_indices(tmp_path, index_dtype):
    X = sp.csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]]))
    X.indices = X.indices.astype(index_dtype)
    X.indptr = X.indptr.astype(index_dtype)
    X.indices.flags.writeable = False
    X.indptr.flags.writeable = False
    path = tmp_path / "read_only.svm"

    svmloader.save_svmfile(str(path), X, np.array([1, 0]))

    assert path.read_text() == "1 2:1.5\n0 1:2\n"


def test_save_svmfile_rejects_mismatched_index_dtypes(tmp_path):
    X = sp.csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]]))
    X.indptr = X.indptr.astype(np.int64)  # indices stay int32
    assert X.indices.dtype == np.int32

    with pytest.raises(TypeError, match="same dtype"):
        svmloader.save_svmfile(str(tmp_path / "out.svm"), X)


def test_save_svmfile_rejects_unsupported_index_dtype(tmp_path):
    X = sp.csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]]))
    X.indices = X.indices.astype(np.int16)
    X.indptr = X.indptr.astype(np.int16)

    with pytest.raises(TypeError, match="unsupported index dtype"):
        svmloader.save_svmfile(str(tmp_path / "out.svm"), X)


def test_save_svmfile_accepts_csr_array(tmp_path):
    if not hasattr(sp, "csr_array"):
        pytest.skip("scipy too old for csr_array")
    X = sp.csr_array(np.array([[0.0, 1.5], [2.0, 0.0]]))
    path = tmp_path / "array.svm"

    svmloader.save_svmfile(str(path), X, np.array([1, 0]))

    assert path.read_text() == "1 2:1.5\n0 1:2\n"


def test_save_svmfile_without_labels_zero_based(tmp_path):
    X = sp.csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]]))
    path = tmp_path / "out.svm"

    svmloader.save_svmfile(str(path), X, zero_based=True)

    assert path.read_text() == "1:1.5\n0:2\n"


def test_save_svmfile_single_precision_data(tmp_path):
    X = sp.csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]], dtype=np.float32))
    path = tmp_path / "float32.svm"

    svmloader.save_svmfile(str(path), X)

    assert path.read_text() == "2:1.5\n1:2\n"


@pytest.mark.parametrize("suffix", [".gz", ".bz2"])
def test_load_compressed_file(tmp_path, suffix):
    plain = tmp_path / "plain.svm"
    plain.write_text(DATA)
    compressed = tmp_path / ("data.svm" + suffix)
    if suffix == ".gz":
        import gzip
        with gzip.open(compressed, "wb") as fp:
            fp.write(DATA.encode())
    else:
        import bz2
        with bz2.open(compressed, "wb") as fp:
            fp.write(DATA.encode())

    X_plain, y_plain = svmloader.load_svmfile(str(plain), zero_based=False)
    X_comp, y_comp = svmloader.load_svmfile(str(compressed), zero_based=False)

    np.testing.assert_array_equal(y_comp, y_plain)
    np.testing.assert_allclose(X_comp.toarray(), X_plain.toarray())


def test_load_crlf_file(tmp_path):
    path = tmp_path / "crlf.svm"
    path.write_bytes(DATA.replace("\n", "\r\n").encode())

    X, y = svmloader.load_svmfile(str(path), zero_based=False)

    assert y.tolist() == [1, 0]
    np.testing.assert_allclose(X.toarray(), [[0.5, 0.0, 2.0], [0.0, 1.0, 0.0]])


def test_save_svmfile_writes_lf_line_endings(tmp_path):
    X = sp.csr_matrix(np.array([[0.0, 1.5], [2.0, 0.0]]))
    path = tmp_path / "out.svm"

    svmloader.save_svmfile(str(path), X, np.array([1, 0]))

    assert path.read_bytes() == b"1 2:1.5\n0 1:2\n"
