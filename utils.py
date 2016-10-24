#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import collections

from keras import backend as K


def ones(shape, dtype=K.floatx()):
    """Return all-ones tensor of given shape and type."""
    # As of Keras version 1.1.0, Keras ones() requires integer values
    # in shape (e.g. calling np.ones() with the Theano backend) and
    # thus can't be called with tensor values. This version avoids the
    # issue by using the backend ones() instead.
    if K.backend() == 'theano':
        from theano import tensor as T
        return T.ones(shape, dtype)
    else:
        assert K.backend() == 'tensorflow'
        import tensorflow as tf
        return tf.ones(shape, dtype)


def zeros(shape, dtype=K.floatx()):
    """Return all-zeros tensor of given shape and type."""
    # As of Keras version 1.1.0, Keras zeros() requires integer values
    # in shape (e.g. calling np.zeros() with the Theano backend) and
    # thus can't be called with tensor values. This version avoids the
    # issue by using the backend zeros() instead.
    if K.backend() == 'theano':
        from theano import tensor as T
        return T.zeros(shape, dtype)
    else:
        assert K.backend() == 'tensorflow'
        import tensorflow as tf
        return tf.zeros(shape, dtype)


def values(value, shape, dtype=K.floatx()):
    """Return tensor of given shape and type filled with given value."""
    return value * ones(shape, dtype)    # or zeros() + ?


def meshgrid(i, j, indexing='ij'):
    """Return matrices broadcasting indices on a 2d grid.

    This is a partial backend-independent version of TensorFlow meshgrid()
    (https://www.tensorflow.org/api_docs/python/array_ops.html#meshgrid)
    with matrix indexing.
    """
    if K.ndim(i) != 1 or K.ndim(j) != 1:
        raise ValueError('need ndim() == 1')
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        I, J = tf.meshgrid(i, j, indexing=indexing)
    else:
        assert K.backend() == 'theano'
        from theano import tensor as T
        I = T.repeat(i, K.shape(j)[0])
        J = T.tile(j, K.shape(i)[0])
    shape = (K.shape(i)[0], K.shape(j)[0])
    return K.reshape(I, shape), K.reshape(J, shape)


def one_hot(a, size=None, dtype=np.int32):
    """Return one-hot representation of given tensor or numpy array."""
    # http://stackoverflow.com/a/37323404
    if isinstance(a, np.ndarray):
        if size is None:
            size = a.max() + 1
        return np.eye(size, dtype=dtype)[a]
    else:
        if size is None:
            raise NotImplementedError()
        return K.eye(size, dtype)[a]


def unique(iterable):
    """Return unique values from iterable."""
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]


def arange(start, stop=None, dtype=None):
    """Keras backend-independent range for tensor values."""
    if stop is None:
        start, stop = 0, start
    if K.backend() == 'theano':
        from theano import tensor as T
        range_ = T.arange(start, stop)
    else:
        assert K.backend() == 'tensorflow'
        import tensorflow as tf
        range_ = tf.range(start, stop)
    if dtype is not None:
        range_ = K.cast(range_, dtype=dtype)
    return range_


def ndim(a):
    """Return the number of dimensions in a tensor or numpy array."""
    if isinstance(a, np.ndarray):
        return a.ndim
    else:
        return K.ndim(a)


def zeros_like(a):
    """Return array of zeros with shape of given tensor or numpy array."""
    if isinstance(a, np.ndarray):
        return np.zeros_like(a)
    else:
        return K.zeros_like(a)


def check_ndim(a, d):
    """Check that number of dimensions in a is d, raise ValueError otherwise."""
    if ndim(a) != d:
        raise ValueError('expected {}d value, got {}d'.format(d, ndim(a)))


def normalize_and_check_ndim(values, d):
    """Convert Python Sequences to numpy array and check that the number
    of dimensions in each given value matches d.
    """
    def normalize(a):
        if isinstance(a, collections.Sequence):
            return np.asarray(a)
        else:
            return a
    values = [normalize(v) for v in values]
    for v in values:
        check_ndim(v, d)
    return values


def outer_product(a, b, batch=False):
    """Outer product of two vectors.

    If batch is True, return batchwise outer product.
    """
    if batch:
        return batch_outer_product(a, b)
    a, b = normalize_and_check_ndim([a, b], 1)
    # The outer product is equivalent to matrix multiplication a * b
    # where the vector a is interpreted as a column matrix and the
    # vector b as a row matrix. The following reshaping and
    # multiplication accomplishes this.
    return a[:, np.newaxis] * b[np.newaxis, :]


def batch_outer_product(a, b):
    """Batchwise outer product of pairs of vectors.

    Expects two 2d tensors of shapes (b, m) and (b, n) and returns a
    3d tensor of shape (b, m, n) where each of the (m, n) submatrices
    is the outer product of corresponding vectors.
    """
    a, b = normalize_and_check_ndim([a, b], 2)
    # This is a batchwise version of the matrix multiplication approach
    # used for outer_product(), see explanation there.
    return a[:, :, np.newaxis] * b[:, np.newaxis, :]


def outer_sum(a, b, batch=False):
    """\"Outer sum" of two vectors.

    If batch is True, return batchwise outer sum.
    """
    if batch:
        return batch_outer_sum(a, b)
    # TODO: naming. Surely this has to be called something sensible?
    a, b = normalize_and_check_ndim([a, b], 1)
    # Due to broadcasting, this sum works analogously to matrix
    # multiplication. See also comments in outer_product().
    return a[:, np.newaxis] + b[np.newaxis, :]


def batch_outer_sum(a, b):
    """Batchwise "outer sum" of pairs of vectors.

    Expects two 2d tensors of shapes (b, m) and (b, n) and returns a
    3d tensor of shape (b, m, n) where each of the (m, n) submatrices
    is the "outer sum" of corresponding vectors.
    """
    a, b = normalize_and_check_ndim([a, b], 2)
    # Due to broadcasting, this sum works analogously to batch matrix
    # multiplication. See also comments in batch_outer_product().
    return a[:, :, np.newaxis] + b[:, np.newaxis, :]


def logsumexp(x, axis=None):
    """Return the log of the sum of exponentials of elements of x.

    Preserves numerical precision around the maximum value by
    initially subtracting and finally adding back in the max.

    See e.g. https://en.wikipedia.org/wiki/LogSumExp ,
    http://math.stackexchange.com/a/648606 .
    """
    xmax = K.max(x, axis=axis, keepdims=True)
    xmax_ = K.max(x, axis=axis)
    return xmax_ + K.log(K.sum(K.exp(x - xmax), axis=axis))


def multi_index(t, indices):
    """Return t[indices] where indices is a sequence.

    This Implements a subset of "fancy indexing" operations such as
    indexing with a tuple (e.g. t[idx1, idx2]) in a way that is
    transparent to the choice of Keras backend. This is needed because
    still as of version 0.11, TensorFlow doesn't fully support
    Numpy/Theano-like advanced indexing (see
    https://github.com/tensorflow/tensorflow/issues/206,
    https://github.com/tensorflow/tensorflow/issues/418,
    https://github.com/tensorflow/tensorflow/issues/4638).
    """
    if K._BACKEND == 'theano':
        return t[tuple(indices)]
        #from operator import getitem
        # Use native Theano indexing. 
        #return getitem(t, tuple(indices))    # Equivalent to t[indices].
    else:
        return _tf_multi_index(t, indices)


def _tf_multi_index(t, indices):
    """Partial TensorFlow implementation of Theano t[indices]."""
    # Note: this is far from a full implementation of Theano fancy
    # indexing, use with care.
    assert K._BACKEND == 'tensorflow'
    from collections import Sequence
    import tensorflow as tf

    if not isinstance(indices, Sequence):
        raise ValueError(indices)

    if len(indices) == 1:
        return tf.gather(t, indices[0])    # gather() suffices for 1d
    if K.ndim(t) == len(indices):
        # Index n-dimensional tensor with n indices: pack the indices
        # from e.g. [[i_0, i_1, ...] [j_0, j_1, ...]] to [[i_0, j_0],
        # [i_1, j_1], ...] and use gather_nd()
        # (https://www.tensorflow.org/api_docs/python/array_ops.html#gather_nd)
        # TODO: check that all i in indices have ndim n-1 
        # TODO: support broadcasting for numpy arrays with np.broadcast_to()
        #indices = tf.pack(list(indices), axis=len(indices)-1)
        indices = tf.pack(list(indices), axis=-1)
        # indices = tf.Print(indices, [indices], 'indices', summarize=100)
        return tf.gather_nd(t, indices)
    else:
        raise NotImplementedError('index {} with {}'.format(t, indices))


def _test():
    # Self-tests. TODO: rewrite using proper testing framework.
    u = [1, 2, 4]
    v = [1, 10, 100]
    assert np.array_equal(outer_product(u, v), np.outer(u, v))

    # Keras tests of outer_product and outer_sum
    u = K.placeholder(ndim=1)
    v = K.placeholder(ndim=1)
    p = outer_product(u, v)
    s = outer_sum(u, v)
    fp = K.function([u, v], [p])
    fs = K.function([u, v], [s])
    x = [1, 2, 4]
    y = [1, 10, 100]
    r = fp([x,y])[0]
    print('outer product: {} x {} = {}'.format(x, y, r))
    r = fs([x,y])[0]
    print('outer sum: {} (+) {} = {}'.format(x, y, r))

    # Keras test of batch_outer_product
    bu = K.placeholder(ndim=2)
    bv = K.placeholder(ndim=2)
    bp = batch_outer_product(bu, bv)
    bs = batch_outer_sum(bu, bv)
    bpf = K.function([bu, bv], [bp])
    bsf = K.function([bu, bv], [bs])
    bx = [[1, 2, 4], [2, 4, 8]]
    by = [[1, 10, 100], [1, 10, 100]]
    br = bpf([bx, by])[0]
    print('batch outer product: {} x {} = {}'.format(bx, by, br))
    br = bsf([bx, by])[0]
    print('batch outer sum: {} (+) {} = {}'.format(bx, by, br))

    # TODO: test multi_index()

if __name__ == '__main__':
    _test()
