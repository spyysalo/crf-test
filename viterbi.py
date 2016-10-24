#!/usr/bin/env python

from __future__ import print_function

import numpy as np

from logging import debug, warn

from keras import backend as K

from utils import outer_sum, multi_index, arange, zeros_like
from scan import scan


def viterbi(observations, transitions, sequence_len, batch=False):
    """Implementation of the Viterbi algorithm in Keras.

    Returns the most likely state sequence for the given observations
    and transitions by first making a forward pass through the
    sequence that calculates the maximum probability of reaching
    each state and stores the index of the previous state on that
    path, and then making a backward pass collecting those indices.

    See e.g. https://en.wikipedia.org/wiki/Viterbi_algorithm .

    Args:
        observations (tensor): A tensor of the observation log
            probabilities, shape (sequence_len, num_states) if 
            batch is False, (batch_size, sequence_len, num_states)
            otherwise.
        transitions (tensor): A (num_states, num_states) tensor of
            the transition weights (log probabilities).
        sequence_len (int): The number of steps in the sequence.
            This must be given because unrolling scan() requires a
            definite (not tensor) value.
        batch (bool): Whether to run in batchwise mode. If True, the
            first dimension of observations corresponds to the batch.

    Returns:
        sequence_len vector of Viterbi path state indices if
        batch is False or (batch_size, sequence_len) matrix of the
        same otherwise.
    """
    # TODO: eliminate K.cast()s if possible
    # TODO: eliminate switches on batch
    # TODO: compare performance with transitions in closure and
    # transitions as non_sequences argument
    forward_step = make_viterbi_forward_step(transitions, batch=batch)

    # index_like is example of the 2nd argument shape, its value is unused.
    if not batch:
        first, rest = observations[0], observations[1:]
        index_like = K.cast(K.zeros_like(first), 'int64')
    else:
        first, rest = observations[:, 0], observations[:, 1:]
        index_like = zeros_like(first)
    sequence_len -= 1    # Exclude first

    (best_scores, best_indices), _ = scan(
        forward_step, rest, [first, index_like], n_steps=sequence_len,
        batch=batch
    )

    batch_size = None if not batch else K.shape(observations)[0]
    backward_step = make_viterbi_backward_step(batch, batch_size)

    if not batch:
        reverse_indices = K.reverse(best_indices, axes=[0])
        last_idx = K.argmax(best_scores[-1])
    else:
        reverse_indices = K.reverse(best_indices, axes=[1])
        last_idx = K.argmax(best_scores[:, -1])

    # TODO: consider go_backwards in scan instead of double reverse
    sequence, _ = scan(
        backward_step, reverse_indices, last_idx, n_steps=sequence_len,
        batch=batch
    )

    if not batch:
        sequence = K.concatenate([sequence[::-1], [last_idx]], axis=0)
    else:
        sequence = K.concatenate([K.reverse(sequence, axes=[1]),
                                  K.expand_dims(last_idx)], axis=1)
    return sequence


def make_viterbi_forward_step(transitions, batch=False):
    """Return scan() step function for the Viterbi algorithm forward pass.

    The returned function is used in viterbi() to recursively
    calculate the maximum probability of reaching each state and the
    index of the previous state on the path leading to that state for
    each point in the input sequence.

    The process is directly analogous to the forward algorithm, but
    replaces the sum by max and has also a second return value (the
    index). See also make_forward_step().

    Args:
        transitions (tensor): A (num_states, num_states) tensor of
            the transition weights (log probabilities).
        batch (bool): Whether to create a function for batchwise
            inputs where the first dimension corresponds to the batch.

    Returns:
        scan() step function calculating the maximum probability for
        reaching each state and the index of the previous state on the
        path there. If batch is False, the function expects
        observation log probabilities as a (sequence_length,
        num_states) tensor, if true, as (batch_size, sequence_length,
        num_states).
    """
    # Note: the previous state index (prev_idx) is not used by step(),
    # but required as part of the scan() output for the backward pass.
    axis = 0 if not batch else 1
    def step(obs, prev, prev_idx):
        scores = outer_sum(prev, obs, batch) + transitions
        return K.max(scores, axis=axis), K.argmax(scores, axis=axis)
    return step


def make_viterbi_backward_step(batch=False, batch_size=None):
    """Return scan() step function for the Viterbi algorithm backward pass.

    The returned function is used in viterbi() to collect the indices of
    the states on the most likely path that were calculated in the forward
    pass.

    Args:
        batch (bool): Whether to create a function for batchwise
            inputs where the first dimension corresponds to the batch.
        batch_size (0d tensor): The batch size if batch is True or
            None otherwise.

    Returns:
        scan() step function collecting indices calculate in the forward
        pass.
    """
    if not batch:
        def step(best_indices, previous_index):
            return best_indices[K.cast(previous_index, 'int32')]
    else:
        def step(best_indices, previous_indices):
            # previous_indices is a batch_size vector of state indices and
            # best_indices a (batch_size, num_states) matrix. Return
            # [best_indices[previous[b]] for b in range(batch_size)].
            b_idx = arange(batch_size, dtype=K.dtype(previous_indices))
            return multi_index(best_indices, [b_idx, previous_indices])
    return step

                   
########## test


if __name__ == '__main__':
    from testdata import transition_log_probabilities
    from testdata import batch_observation_log_probabilities

    batch_size = len(batch_observation_log_probabilities)
    n_steps = len(batch_observation_log_probabilities[0])

    observations = K.placeholder(ndim=2, name='observations')
    transitions = K.placeholder(ndim=2, name='transitions')
    result = viterbi(observations, transitions, n_steps)
    f = K.function([observations, transitions], [result])

    print('single:')
    for observation_log_probabilities in batch_observation_log_probabilities:
        print(f([observation_log_probabilities,
                 transition_log_probabilities])[0])

    batch_observations = K.placeholder(ndim=3, name='batch_observations')
    batch_result = viterbi(batch_observations, transitions, n_steps,
                           batch=True)

    bf = K.function([batch_observations, transitions], [batch_result])
    br = bf([batch_observation_log_probabilities,
             transition_log_probabilities])[0]
    print('batch:')
    for r in br:
        print(r)
