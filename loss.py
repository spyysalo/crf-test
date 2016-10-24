#!/usr/bin/env python

import numpy as np

from keras import backend as K

import theano
import theano.tensor as T

from utils import multi_index, arange, zeros, values, meshgrid
from forward import forward

log_0 = -1000    # Small log prob representing zero probability

np.random.seed(1234)    # make initialization repeatable


def crf_loss(true_indices, observations, transitions, sequence_len,
             batch=False):
    """
    Linear-chain conditional random field (CRF) loss function.
    
    Args:
        true_indices (tensor): A tensor of the correct sequence of
            state indices, shape (sequence_len, ) if batch is False,
            (batch_size, sequence_len) otherwise.
        observations (tensor): A tensor of the observation log
            probabilities, shape (sequence_len, num_states) if 
            batch is False, (batch_size, sequence_len, num_states)
            otherwise.
        transitions (tensor): A (num_states, num_states) tensor of
            the transition weights (log probabilities).
        sequence_len (int): The number of steps in the sequence.
            This must be given because unrolling scan() requires a
            definite (not tensor) value.
        batch (bool): Whether to run in batchwise mode.

    Returns:
        loss tensor, 0d if batch is False, batch_size vector
        otherwise.
    """
    if K.ndim(observations) != 2 + int(batch):
        raise ValueError('ndim(observations) == {}'.format(K.ndim(observations)))

    num_states = K.shape(observations)[int(batch) + 1]
    observations = pad_scores(observations, batch=batch)
    true_indices = pad_indices(true_indices, num_states, batch=batch)

    if not batch:
        # Index observations by sequence position [0 1 2 ...] and true
        # index, and transitions by consecutive values from true_indices.
        obs_index = [np.arange(sequence_len+2), true_indices]
        from_idx, to_idx = true_indices[:-1], true_indices[1:]
    else:
        # Index observations by batch, sequence position, and true index:
        # i_idx = [ [ 0 0 0 ... ] [ 1 1 1 ... ] ... ]
        # j_idx = [ [ 0 1 2 ... ] [ 0 1 2 ... ] ... ]
        # Index transitions by consecutive values from true_indices.
        batch_size = K.shape(observations)[0]
        i_idx, j_idx = meshgrid(arange(batch_size), arange(sequence_len+2))
        obs_index = [i_idx, j_idx, true_indices]
        from_idx, to_idx = true_indices[:, :-1], true_indices[:, 1:]
    
    true_obs_scores = multi_index(observations, obs_index)
    true_trans_scores = multi_index(transitions, [from_idx, to_idx])

    true_path_score = (K.sum(true_obs_scores, axis=int(batch)) +
                       K.sum(true_trans_scores, axis=int(batch)))
    all_paths_scores = forward(observations, transitions, sequence_len+2,
                               batch=batch)
    return all_paths_scores - true_path_score


def pad_scores(scores, batch=False):
    """Return score matrix extended to include start and end states.

    Extends the (sequence_len, num_states) (sub)matrix of observation
    log probabilities to

        0 0 ... 0 1 0
        X X ... X 0 0
        X X ... X 0 0
            ...
        X X ... X 0 0
        0 0 ... 0 0 1

    where the Xs are the original values, the 0s are a log-space
    equivalent of zero probability, and the 1s the log of probability
    one (i.e. 0). If batch is True, this padding is added for each
    item in the batch.

    This corresponds to a probability of one for the new start state
    at the beginning of the sequence, a probability of one for the new
    end state at its end, and otherwise unmodified probabilities.

    Use together with batch_pad_indices().
    """
    if K.ndim(scores) != 2 + int(batch):
        raise ValueError('ndim(scores) == {}'.format(K.ndim(scores)))

    batch_shape = () if not batch else (K.shape(scores)[0], )

    sequence_len = K.shape(scores)[int(batch)+0]
    num_states = K.shape(scores)[int(batch)+1]
    dtype = K.dtype(scores)

    log_0s = lambda *shape: values(log_0, batch_shape+tuple(shape), dtype)
    log_1s = lambda *shape: values(0, batch_shape+tuple(shape), dtype)
    cat = lambda *values, **kwargs: K.concatenate(list(values), **kwargs)

    new_cols = log_0s(sequence_len, 2)
    start_row = cat(log_0s(1, num_states), log_1s(1, 1), log_0s(1, 1))
    end_row = cat(log_0s(1, num_states+1), log_1s(1, 1))

    return cat(
        start_row,
        cat(scores, new_cols),
        end_row, axis=len(batch_shape)
    )


def pad_indices(indices, num_states, batch=False):
    """Return sequence of indices extended to include start and end states.

    Extends the indices for each sequence_len vector (or submatrix) to

        N X X ... X N+1

    where the Xs are the original indices and N is the original number
    of states. If batch is True, this padding is added for each item
    in the batch.

    This corresponds to a sequence that starts with a new state with
    index N, ends with a new state with index N+1, and is otherwise
    unmodified.
    
    Use together with pad_scores().
    """
    if K.ndim(indices) != 1 + int(batch):
        raise ValueError('ndim(indices) == {}'.format(K.ndim(indices)))

    batch_shape = () if not batch else (K.shape(indices)[0], )
    dtype = K.dtype(indices)

    start_idx = values(num_states, batch_shape + (1,), dtype)
    end_idx = values(num_states+1, batch_shape + (1,), dtype)
    return K.concatenate([start_idx, indices, end_idx], axis=len(batch_shape))


def make_transitions(num_states):
    """Return randomly initialized transition matrix."""
    # Assume that start and end states will be added to given number.
    # (See also comment in pad_scores().)
    num_states += 2
    shape = (num_states, num_states)
    # TODO use Keras initialization
    drange = np.sqrt(6. / (np.sum(shape)))
    value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return K.variable(value, dtype=K.floatx(), name='transitions')


########## test


if __name__ == '__main__':
    from testdata import batch_observation_log_probabilities as batch_obs_logprob
    from testdata import batch_true_indices

    batch_size = len(batch_obs_logprob)
    sequence_len = len(batch_obs_logprob[0])
    num_states = len(batch_obs_logprob[0][0])

    indices = K.placeholder(shape=(sequence_len, ), dtype='int32')
    scores = K.placeholder(shape=(sequence_len, num_states), dtype=K.floatx())
    transitions = make_transitions(num_states)
    loss = crf_loss(indices, scores, transitions, sequence_len)

    loss_f = K.function([indices, scores], [loss])

    print('single:')
    for obs_logprob, true_indices in zip(batch_obs_logprob,
                                         batch_true_indices):
        print('{:.3f}'.format(float(loss_f([true_indices, obs_logprob])[0])))

    batch_indices = K.placeholder(shape=(batch_size, sequence_len, ),
                                  dtype='int32')
    batch_scores = K.placeholder(shape=(batch_size, sequence_len, num_states),
                                 dtype=K.floatx())
    batch_loss = crf_loss(batch_indices, batch_scores, transitions, 
                          sequence_len, batch=True)
    
    loss_bf = K.function([batch_indices, batch_scores], [batch_loss])

    batch_r = loss_bf([batch_true_indices, batch_obs_logprob])[0]
    print('batch:')
    for r in batch_r:
        print('{:.3f}'.format(r))
