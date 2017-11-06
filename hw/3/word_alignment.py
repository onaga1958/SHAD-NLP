#!/usr/bin/python

import sys
import numpy as np
from models import TranslationModel
from models import TransitionModel
from models import PriorModel
from utils import read_all_tokens, output_alignments_per_test_set
from hmm import backward, forward, viterby
from nltk.stem import WordNetLemmatizer


def get_alignment_posteriors(src_tokens, trg_tokens, transition_model, translation_model):
    "Compute the posterior alignment probability p(a_j=i | f, e) for each target token f_j."
    if isinstance(transition_model, TransitionModel):
        initial, transition = transition_model.get_parameters_for_sentence_pair(len(src_tokens))
        translation = translation_model.get_parameters_for_sentence_pair(src_tokens, trg_tokens)

        posteriors = np.zeros((len(trg_tokens) - 1, len(src_tokens), len(src_tokens)))
        single_posteriors = np.zeros((len(trg_tokens), len(src_tokens)))

        params = (initial, transition, translation)
        observations = np.arange(len(trg_tokens))
        alpha = forward(params, observations)
        beta = backward(params, observations)
        answers = viterby(*params)

        for t in range(len(trg_tokens) - 1):
            nominator = (alpha[t, :] * transition.T).T * translation[:, t + 1] * beta[t + 1, :]
            posteriors[t] = nominator / np.sum(nominator)

        nominator = alpha * beta
        single_posteriors = (nominator.T / np.sum(nominator, axis=1)).T

        log_likelihood = (np.log(initial[answers[0]]) +
                          np.sum(np.log(transition[answers[:-1], answers[1:]])) +
                          np.sum(np.log(translation[answers, np.arange(len(trg_tokens))])))
        return (posteriors, single_posteriors), log_likelihood, answers
    else:
        # here transition_model is a prior_model
        prior = transition_model.get_parameters_for_sentence_pair(len(src_tokens),
                                                                  len(trg_tokens))
        traslation = translation_model.get_parameters_for_sentence_pair(src_tokens,
                                                                        trg_tokens)

        nominator = prior * traslation
        denominator = np.sum(nominator, axis=0)
        alignment_posteriors = nominator / denominator

        answers = np.argmax(alignment_posteriors, axis=0)
        arange = np.arange(len(trg_tokens))
        log_likelihood = (np.log(prior[answers, arange]).sum() +
                          np.log(traslation[answers, arange]).sum())

        return [len(trg_tokens), alignment_posteriors.T], log_likelihood, answers


def collect_expected_statistics(src_corpus, trg_corpus, transition_model, translation_model):
    "E-step: infer posterior distribution over each sentence pair and collect statistics."
    corpus_log_likelihood = 0.0
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        # Infer posterior
        alignment_posteriors, log_likelihood, _ = get_alignment_posteriors(src_tokens, trg_tokens, transition_model,
                                                                           translation_model)
        # Collect statistics in each model.
        transition_model.collect_statistics(len(src_tokens), *alignment_posteriors)
        translation_model.collect_statistics(src_tokens, trg_tokens, alignment_posteriors[1])
        # Update log prob
        corpus_log_likelihood += log_likelihood
    return corpus_log_likelihood


def estimate_models(src_corpus, trg_corpus, transition_model, translation_model, num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        # E-step
        corpus_log_likelihood = collect_expected_statistics(src_corpus, trg_corpus, transition_model,
                                                            translation_model)
        # M-step
        transition_model.recompute_parameters()
        translation_model.recompute_parameters()
        if iteration > 0:
            print(str(iteration) + ": corpus log likelihood: %1.3f" % corpus_log_likelihood)
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

            sentence[i] = word[:5]

    return corpus


if __name__ == "__main__":
    if not len(sys.argv) == 6:
        print("Usage ./word_alignment.py src_corpus trg_corpus pretrain_iteratioins iterations " +
              "output_prefix.")
        sys.exit(0)
    src_corpus = read_all_tokens(sys.argv[1])
    trg_corpus = read_all_tokens(sys.argv[2])

    src_corpus = normalize(src_corpus)
    trg_corpus = normalize(trg_corpus, sys.argv[2].find('lemmas') == -1)

    IBM1_num_iterations = int(sys.argv[3])
    num_iterations = int(sys.argv[4])
    output_prefix = sys.argv[5]
    assert len(src_corpus) == len(trg_corpus), "Corpora should be same size!"

    transition_model, translation_model = initialize_models(src_corpus,
                                                            trg_corpus)
    prior_model = PriorModel()
    _, translation_model = estimate_models(src_corpus,
                                           trg_corpus,
                                           prior_model,
                                           translation_model,
                                           IBM1_num_iterations)
    transition_model, translation_model = estimate_models(src_corpus,
                                                          trg_corpus,
                                                          transition_model,
                                                          translation_model,
                                                          num_iterations)
    alignments = align_corpus(src_corpus, trg_corpus, transition_model,
                              translation_model)
    output_alignments_per_test_set(alignments, output_prefix)
