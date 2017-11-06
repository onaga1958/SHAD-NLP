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


class Model:
    def __init__(self, src_corpus, trg_corpus):
        self._probs = defaultdict(lambda: defaultdict(lambda: 1))
        self._reset_cnt()

    def _reset_cnt(self):
        self._counter = defaultdict(lambda: defaultdict(lambda: 0))

    def recompute_parameters(self):
        for first_key, inner_dict in self._counter.items():
            total_count = sum(list(inner_dict.values()))
            for second_key, cnt in inner_dict.items():
                self._probs[first_key][second_key] = cnt / total_count

        self._reset_cnt()


class TranslationModel(Model):
    "Models conditional distribution over trg words given a src word."
    def __init__(self, src_corpus, trg_corpus):
        super().__init__(src_corpus, trg_corpus)
        for s_sent, t_sent in zip(src_corpus, trg_corpus):
            for s_word in s_sent:
                for t_word in t_sent:
                    self._counter[s_word][t_word] += 1
        self.recompute_parameters()

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token."
        return self._probs[src_token][trg_token]

    def get_parameters_for_sentence_pair(self, src_tokens, trg_tokens):
        "Return numpy array with t[i][j] = p(f_j|e_i)."
        return np.array([[self.get_conditional_prob(src_token, trg_token)
                          for trg_token in trg_tokens] for src_token in src_tokens])

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate counts of translations from posterior_matrix[j][i] = p(a_j=i|e, f)"
        for i, src_token in enumerate(src_tokens):
            for j, trg_token in enumerate(trg_tokens):
                self._counter[src_token][trg_token] += posterior_matrix[j, i]


class TransitionModel(Model):
    "Models the prior probability of an alignment given the previous token's alignment."

    def get_parameters_for_sentence_pair(self, src_length):
        "Retrieve the parameters for this sentence pair: A[k, i] = p(a_{j} = i|a_{j-1} = k)"
        transition = np.array([[self._probs[(src_length)][i - k] for i in range(src_length)]
                               for k in range(src_length)])
        initial = np.array([self._probs[(src_length, 'initial')][i] for i in range(src_length)])
        return initial, transition

    def collect_statistics(self, src_length, posteriors, single_posteriors):
        "Accumulate statistics from transition_posteriors[k][i]: p(a_{j} = i, a_{j-1} = k|e, f)"
        for i in range(src_length):
            self._counter[(src_length, 'initial')][i] += single_posteriors[0, i]
            for k in range(src_length):
                self._counter[(src_length)][i - k] += posteriors[:, k, i].sum()


class PriorModel:
    """
    Models the prior probability of an alignment given only the sentence
    lengths and token indices.
    """

    def __init__(self):
        "Add counters and parameters here for a prior model."
        pass

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a prior probability based on src and trg indices."
        return 1 / src_length

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
        pass

    def recompute_parameters(self, *args):
        pass
