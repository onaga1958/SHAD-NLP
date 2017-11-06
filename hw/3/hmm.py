# Example HMM implementation
#
# The python code is shamelessly stolen from Trevor Cohn.
# Original: http://people.eng.unimelb.edu.au/tcohn/comp90042/HMM.py
#
# This is a generic and very readable implementation of a discrete HMM but it
# will need adapting to do word alignment.
#
# 1. This implementation assumes a fixed number of hidden states across all sequences.
# In word alignment, however, the number of hidden states is equal to the number of
# source tokens so your HMM parameters (pi, O, A) will change for each sentence.
#
# 2. This implementation is very readable but not very efficient or practical.
# (i) You should make it more efficent by collapsing loops into numpy matrix operations.
# (ii) You should scale the forward and backward probabilities (i.e. alpha and beta) to
# avoid numerical underflow on longer sequences.
#
# 3. You will probably only need the forward and backward methods from this file.
# Once you have the alpha and beta probabilities you can easily compute the statistics
# needed by the models in models.py.

import numpy as np


def forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    alpha = np.zeros((N, S))

    # base case
    alpha[0] = pi * O[:, observations[0]]

    # recursive case
    for i in range(1, N):
        alpha[i] = np.sum(A.T * alpha[i - 1], axis=1) * O[:, observations[i]]
    return alpha


def backward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    beta = np.zeros((N, S))

    # base case
    beta[N - 1] = 1

    # recursive case
    for i in range(N - 2, -1, -1):
        beta[i] = np.sum(A * beta[i + 1] * O[:, observations[i + 1]], axis=1)
    return beta


def viterby(pi, A, O):
    probs = np.zeros_like(O)
    max_path = np.zeros_like(probs, dtype=np.int8)
    for t in range(O.shape[1]):
        if t:
            possible_probs = probs[:, t - 1] + (np.log(O[:, t]) + np.log(A)).T
            probs[:, t] = np.max(possible_probs, axis=1)
            max_path[:, t] = np.argmax(possible_probs, axis=1)
        else:
            probs[:, t] = np.log(pi) + np.log(O[:, t])

    result = [np.argmax(probs[:, -1])]
    for i in range(O.shape[1] - 1, 0, -1):
        result.append(max_path[result[-1]][i])
    return np.array(list(reversed(result)))
