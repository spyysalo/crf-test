#!/usr/bin/env python

# This is a copy of code from Guillaume Lample's tagger with minor
# changes, used as reference to cross-check the Keras version.
# Original available from https://github.com/glample/tagger (MIT license)

from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

np.random.seed(1234)    # make shared() repeatable


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


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


def crf_cost(tag_ids, tags_scores, n_tags, s_len):
    # tag_ids: gold indices (int32 vector)
    # tags_scores: input predictions (log prob) (float64 matrix)
    # n_tags: number of different states (int)
    # s_len: sequence / sentence length (int64 scalar)
    transitions = shared((n_tags + 2, n_tags + 2), 'transitions')

    small = -1000
    b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
    e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
    observations = T.concatenate(
        [tags_scores, small * T.ones((s_len, 2))],
        axis=1
    )
    observations = T.concatenate(
        [b_s, observations, e_s],
        axis=0
    )

    # Score from tags
    real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

    # Score from transitions
    b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
    e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
    padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
    real_path_score += transitions[
        padded_tags_ids[T.arange(s_len + 1)],
        padded_tags_ids[T.arange(s_len + 1) + 1]
    ].sum()

    all_paths_scores = forward(observations, transitions)
    cost = - (real_path_score - all_paths_scores)
    return cost


if __name__ == '__main__':
    from testdata import batch_observation_log_probabilities
    from testdata import batch_true_indices

    n_tags = len(batch_observation_log_probabilities[0][0])
    s_len = len(batch_observation_log_probabilities[0])

    tag_ids = T.ivector()
    tags_scores = T.dmatrix()
    cost = crf_cost(tag_ids, tags_scores, n_tags, s_len)

    cost_f = theano.function([tag_ids, tags_scores], cost)

    # This implementation doesn't do batching, so just run the same thing
    # twice with different heading to compare output to one that does.
    for s in ('single:', 'batch:'):
        print(s)
        for i in range(len(batch_observation_log_probabilities)):
            v = float(cost_f(batch_true_indices[i],
                             batch_observation_log_probabilities[i]))
            print('{:.3f}'.format(v))
