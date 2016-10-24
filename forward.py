#!/usr/bin/env python

from __future__ import print_function

from keras import backend as K

from math import exp

from utils import outer_sum, logsumexp
from scan import scan


def forward(observations, transitions, sequence_len, batch=False):
    """Implementation of the forward algorithm in Keras.

    Returns the log probability of the given observations and transitions
    by recursively summing over the probabilities of all paths through
    the state space. All probabilities are in logarithmic space.

    See e.g. https://en.wikipedia.org/wiki/Forward_algorithm .

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
        Total log probability if batch is False or vector of log
        probabiities otherwise.
    """
    step = make_forward_step(transitions, batch)
    if not batch:
        first, rest = observations[0, :], observations[1:, :]
    else:
        first, rest = observations[:, 0, :], observations[:, 1:, :]
    sequence_len -= 1    # exclude first
    outputs, _ = scan(step, rest, first, n_steps=sequence_len, batch=batch)

    if not batch:
        last, axis = outputs[sequence_len-1], 0
    else:
        last, axis = outputs[:, sequence_len-1], 1
    return logsumexp(last, axis=axis)


def make_forward_step(transitions, batch=False):
    """Return scan() step function for the forward algorithm.

    The returned function is used in forward() to recursively
    calculate the forward variables holding the probability of
    observations with given transition probabilities up to each point
    in the input sequence.

    The calculation begins at a start state and then recursively adds
    up the 1) probabilities from previous states, 2) transitions to
    the state, and 3) the observation probability for the state. 
    All probabilities are in logarithmic space.

    Args:
        transitions (tensor): A (num_states, num_states) tensor of
            the transition weights (log probabilities).
        batch (bool): Whether to create a function for batchwise
            inputs where the first dimension corresponds to the batch.

    Returns:
        scan() step function for calculating the total probability
        summing over all paths through the model. If batch is
        False, the function expects observation log probabilities
        as a (sequence_len, num_states) tensor, if true, as
        (batch_size, sequence_len, num_states).
    """
    axis = 0 if not batch else 1
    def step(obs, prev):
        # The "outer sum" of the previous log-probabilities and the
        # observation log-probabilities gives a (num_states,
        # num_states) matrix of the paired log-probabilities, and
        # adding in the transition log-probabilities gives the matrix
        # representing the log-probabilities of moving from each state
        # to each other and generating the observation. Summing these
        # (in the standard [0,1] probability space, not the log space)
        # along the 1st axis then gives the vector with the total
        # log-probability of being in each state after the step.
        return logsumexp(outer_sum(prev, obs, batch) + transitions, axis=axis)
    return step


########## test


if __name__ == '__main__':
    from testdata import transition_log_probabilities
    from testdata import batch_observation_log_probabilities

    n_steps = len(batch_observation_log_probabilities[0])

    observations = K.placeholder(ndim=2, name='observations')
    transitions = K.placeholder(ndim=2, name='transitions')
    scores = forward(observations, transitions, n_steps)
    forward_f = K.function([observations, transitions], [scores])

    print('single:')
    for observation_log_probabilities in batch_observation_log_probabilities:
        v = float(forward_f([observation_log_probabilities,
                             transition_log_probabilities])[0])
        print('{:.4}'.format(v))

    batch_observations = K.placeholder(ndim=3, name='batch_observations')
    batch_scores = forward(batch_observations, transitions, n_steps, batch=True)
    forward_bf = K.function([batch_observations, transitions], [batch_scores])

    batch_r = forward_bf([batch_observation_log_probabilities,
                          transition_log_probabilities])[0]
    print('batch:')
    for v in batch_r:
        print('{:.4}'.format(float(v)))
