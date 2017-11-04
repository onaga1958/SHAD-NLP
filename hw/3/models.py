# Models for word alignment.
#
# This file contains stubs for three models to use to model word alignments.
# Notation: i = src_index, I = src_length, j = trg_index, J = trg_length.
#
# (i) TranslationModel models p(f|e).
# (ii) PriorModel models p(i|j, I, J).
# (iii) TransitionModel models p(a_{j} = i|a_{j-1} = k).
#
# Each model stores parameters (probabilities) and statistics (counts) as has:
# (i) A method to access a single probability: get_xxx_prob(...).
# (ii) A method to get all probabilities for a sentence pair as a numpy array:
# get_parameters_for_sentence_pair(...).
# (iii) A method to accumulate 'fractional' counts: collect_statistics(...).
# (iv) A method to recompute parameters: recompute_parameters(...).

import numpy as np
from collections import defaultdict


class TranslationModel:
    "Models conditional distribution over trg words given a src word."

    def __init__(self, src_corpus, trg_corpus):
        self._trg_given_src_probs = defaultdict(lambda:
                                                defaultdict(lambda: 1.0))
        self._reset_cnt()

    def _reset_cnt(self):
        self._src_trg_counts = defaultdict(lambda: defaultdict(lambda: 0.0))

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token."
        return self._trg_given_src_probs[src_token][trg_token]

    def get_parameters_for_sentence_pair(self, src_tokens, trg_tokens):
        "Return numpy array with t[i][j] = p(f_j|e_i)."
        return np.array([[self.get_conditional_prob(src_token, trg_token)
                          for trg_token in trg_tokens]
                         for src_token in src_tokens])

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        """
        Accumulate counts of translations from
        posterior_matrix[j][i] = p(a_j=i|e, f)
        """
        for j, src_token in enumerate(src_tokens):
            for i, trg_token in enumerate(trg_tokens):
                add = posterior_matrix[j][i]
                self._src_trg_counts[src_token][trg_token] += add

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        for src_token, trg_dict in self._src_trg_counts.items():
            total_cnt = sum(list(trg_dict.values()))
            for trg_token, cnt in trg_dict.items():
                prob = cnt / total_cnt
                self._trg_given_src_probs[src_token][trg_token] = prob

        self._reset_cnt()


class PriorModel:
    """
    Models the prior probability of an alignment given only the sentence
    lengths and token indices.
    """

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for a prior model."
        self._src_corpus = src_corpus
        self._trg_corpus = trg_corpus
        self._probs = defaultdict(lambda: defaultdict(lambda: 1))
        self._reset_cnt()

    def _reset_cnt(self):
        self._counter = defaultdict(lambda: defaultdict(lambda: 0))

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a prior probability based on src and trg indices."
        return self._probs[(src_length, trg_length, trg_index)][src_index]

    def get_parameters_for_sentence_pair(self, src_length, trg_length):
        "Return a numpy array with all prior p[i][j] = p(i|j, I, J)."
        return np.array([[self.get_prior_prob(i, j, src_length, trg_length)
                          for j in range(trg_length)]
                         for i in range(src_length)])

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        """
        Accumulate counts of alignment events from posterior_matrix[j][i] =
        p(a_j=i|e, f)
        """
        for i in range(src_length):
            for j in range(trg_length):
                cnt = posterior_matrix[i][j]
                self._counter[(src_length, trg_length, j)][i] += cnt

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        for first_key, trg_dict in self._counter.items():
            total_cnt = sum(list(trg_dict.values()))
            for second_key, cnt in trg_dict.items():
                self._probs[first_key][second_key] = cnt / total_cnt

        self._reset_cnt()


class TransitionModel:
    """
    Models the prior probability of an alignment given the previous token's
    alignment.
    """

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for modeling alignment transitions."
        pass

    def get_parameters_for_sentence_pair(self, src_length):
        """
        Retrieve the parameters for this sentence pair:
        A[k, i] = p(a_{j} = i|a_{j-1} = k)
        """
        pass

    def collect_statistics(self, src_length, transition_posteriors):
        """
        Accumulate statistics from transition_posteriors[k][i]:
        p(a_{j} = i, a_{j-1} = k|e, f)
        """
        pass

    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass
