#!/usr/bin/python

import sys
import numpy as np
# from models import PriorModel  # <-- Implemented as a uniform distribution.
from models import TranslationModel  # <-- Not implemented
from models import TransitionModel  # <-- You will need this for an HMM.
from utils import read_all_tokens, output_alignments_per_test_set
from nltk.stem import WordNetLemmatizer
from hmm import forward, backward, viterby


def get_alignment_posteriors(src_tokens, trg_tokens, transition_model,
                             translation_model):
    """
    Compute the posterior alignment probability p(a_j=i | f, e) for each target
    token f_j.
    """
    params = [len(src_tokens)]
    transition = transition_model.get_parameters_for_sentence_pair(*params)
    params = [src_tokens, trg_tokens]
    translation = translation_model.get_parameters_for_sentence_pair(*params)

    params = (transition[0], transition[1:], translation)
    observations = np.arange(len(trg_tokens))
    alpha = forward(params, observations)[0]
    beta = backward(params, observations)[0]

    posterior = np.zeros((len(trg_tokens), len(src_tokens) + 1,
                          len(src_tokens)))
    nominator = alpha * beta
    posterior[:, 0] = (nominator.T / np.sum(nominator, axis=1)).T
    for t in range(1, len(trg_tokens)):
        nominator = (transition[1:] * alpha[t - 1] * translation[:, t] *
                     beta[t])
        posterior[t, 1:] = nominator / np.sum(nominator)

    answers = viterby(*params)

    prev = np.concatenate((np.array([0]), answers[:-1] + 1))
    log_likelihood = np.sum(np.log(transition[prev, answers]) +
                            np.log(translation[answers,
                                               np.arange(len(answers))]))
    return posterior, log_likelihood, answers


def collect_expected_statistics(src_corpus, trg_corpus, transition_model,
                                translation_model):
    """
    E-step: infer posterior distribution over each sentence pair and collect
    statistics.
    """
    corpus_log_likelihood = 0.0
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        # Infer posterior
        result = get_alignment_posteriors(src_tokens, trg_tokens,
                                          transition_model,
                                          translation_model)
        alignment_posteriors, log_likelihood, _ = result
        # Collect statistics in each model.
        transition_model.collect_statistics(len(src_tokens),
                                            alignment_posteriors)
        translation_model.collect_statistics(src_tokens, trg_tokens,
                                             alignment_posteriors)
        # Update log prob
        corpus_log_likelihood += log_likelihood
    return corpus_log_likelihood


def estimate_models(src_corpus, trg_corpus, transition_model, translation_model,
                    num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        # E-step
        corpus_log_likelihood = collect_expected_statistics(src_corpus,
                                                            trg_corpus,
                                                            transition_model,
                                                            translation_model)
        # M-step
        transition_model.recompute_parameters()
        translation_model.recompute_parameters()
        if iteration > 0:
            print(str(iteration) + ' iteration: ' +
                  "corpus log likelihood: %1.3f" % corpus_log_likelihood)
    return transition_model, translation_model


def align_corpus(src_corpus, trg_corpus, transition_model, translation_model):
    "Align each sentence pair in the corpus in turn."
    return [get_alignment_posteriors(src_tokens,
                                     trg_tokens,
                                     transition_model,
                                     translation_model)[2]
            for src_tokens, trg_tokens in zip(src_corpus, trg_corpus)]


def initialize_models(src_corpus, trg_corpus):
    transition_model = TransitionModel(src_corpus, trg_corpus)
    translation_model = TranslationModel(src_corpus, trg_corpus)
    return transition_model, translation_model


def normalize(corpus, not_lemmatized=True):
    lemmatizer = WordNetLemmatizer()
    for sentence in corpus:
        for i, word in enumerate(sentence):
            word = word.lower()
            if not_lemmatized:
                word = lemmatizer.lemmatize(word)

            sentence[i] = word

    return corpus


if __name__ == "__main__":
    if not len(sys.argv) == 5:
        print("Usage ./word_alignment.py src_corpus trg_corpus iterations " +
              "output_prefix.")
        sys.exit(0)
    src_corpus = read_all_tokens(sys.argv[1])
    trg_corpus = read_all_tokens(sys.argv[2])

    src_corpus = normalize(src_corpus)
    trg_corpus = normalize(trg_corpus, sys.argv[2].find('lemmas') == -1)
    num_iterations = int(sys.argv[3])
    output_prefix = sys.argv[4]
    assert len(src_corpus) == len(trg_corpus), "Corpora should be same size!"

    transition_model, translation_model = initialize_models(src_corpus,
                                                            trg_corpus)
    transition_model, translation_model = estimate_models(src_corpus,
                                                          trg_corpus,
                                                          transition_model,
                                                          translation_model,
                                                          num_iterations)
    alignments = align_corpus(src_corpus, trg_corpus, transition_model,
                              translation_model)
    output_alignments_per_test_set(alignments, output_prefix)
