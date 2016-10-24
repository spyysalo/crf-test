#!/usr/bin/env python

# This is a copy of code from Guillaume Lample's tagger with minor
# changes, used as reference to cross-check the Keras version.
# Original available from https://github.com/glample/tagger (MIT license)

from __future__ import print_function

import theano
import theano.tensor as T

from math import exp


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def forward(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1]), 'int32'),
            sequences=T.cast(alpha[1][::-1], 'int32')
        )
        sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1])]])
        return sequence
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)


if __name__ == '__main__':
    from testdata import transition_log_probabilities
    from testdata import batch_observation_log_probabilities

    n_steps = len(batch_observation_log_probabilities[0])

    observations = T.dmatrix('observations')
    transitions = T.dmatrix('transitions')
    result = forward(observations, transitions)
    forward_f = theano.function([observations, transitions], result)
    
    # This implementation doesn't do batching, so just run the same thing
    # twice with different heading to compare output to one that does.
    for s in ('single:', 'batch:'):
        print(s)
        for i in range(len(batch_observation_log_probabilities)):
            v = forward_f(batch_observation_log_probabilities[i],
                          transition_log_probabilities)
            print('{:.4}'.format(float(v)))
