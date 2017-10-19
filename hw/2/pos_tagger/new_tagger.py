#!/usr/bin/env python3
from collections import defaultdict, namedtuple
from math import ceil
from random import shuffle
import numpy as np
import os
import cityhash
import argparse
import sys
import pprint
import pickle
import re


###############################################################################
#                                                                             #
#                                INPUT DATA                                   #
#                                                                             #
###############################################################################


def read_tags(path):
    """
    Read a list of possible tags from file and return the list.
    """
    with open(path) as tag_file:
        lines = tag_file.readlines()
    return [line.strip() for line in lines]


# Word: str
# Sentence: list of str
TaggedWord = namedtuple('TaggedWord', ['text', 'tag'])
# TaggedSentence: list of TaggedWord
# Tags: list of TaggedWord
# TagLattice: list of Tags


def read_tagged_sentences(path):
    """
    Read tagged sentences from file and return array of TaggedSentence.
    """
    with open(path) as sent_file:
        lines = sent_file.readlines()

    result = []
    sent = []
    for line in lines:
        if len(line) == 1:
            result.append(sent)
            sent = []
        elif line[0] != '#':
            line = line.split('\t')
            word = line[1].strip()
            tag = line[3].strip()
            sent.append(TaggedWord(word, tag))

    return result


def write_tagged_sentence(tagged_sentence, f):
    """
    Write tagged sentence to file-like object f.
    """
    for i, tagged_word in enumerate(tagged_sentence):
        f.write('{}\t{}\t\t{}\n'.format(i + 1, tagged_word.text,
                                        tagged_word.tag))
    f.write('\n')


TaggingQuality = namedtuple('TaggingQuality', ['acc'])


def tagging_quality(ref, out):
    """
    Compute tagging quality and reutrn TaggingQuality object.
    """
    nwords = 0
    ncorrect = 0
    import itertools
    for ref_sentence, out_sentence in itertools.zip_longest(ref, out):
        for ref_word, out_word in itertools.zip_longest(ref_sentence,
                                                        out_sentence):
            nwords += 1
            if ref_word.tag == out_word.tag:
                ncorrect += 1

    return ncorrect / nwords


###############################################################################
#                                                                             #
#                             VALUE & UPDATE                                  #
#                                                                             #
###############################################################################


class Value:
    """
    Dense object that holds parameters.
    """

    def __init__(self, n):
        self.values = np.ones(n)

    def __len__(self):
        return len(self.values)

    def dot(self, update):
        result = 0
        for pos, val in zip(update.positions, update.values):
            result += val * self.values[pos]
        return result

    def assign(self, other):
        """
        self = other
        other is Value.
        """
        self.values = other.values

    def assign_mul(self, coeff):
        """
        self = self * coeff
        coeff is float.
        """
        self.values *= coeff

    def assign_madd(self, x, coeff):
        """
        self = self + x * coeff
        x can be either Value or Update.
        coeff is float.
        """
        if isinstance(x, Value):
            self.values += x.values * coeff
        elif isinstance(x, Update):
            for pos, val in zip(x.positions, x.values):
                self.values[pos] += val * coeff
        else:
            raise TypeError('x must be Value or Update')


class Update:
    """
    Sparse object that holds an update of parameters.
    """

    def __init__(self, positions=[], values=[]):
        """
        positions: array of int
        values: array of float
        """
        self.positions = np.array(positions, dtype=np.int64)
        self.values = np.array(values)

    def assign_mul(self, coeff):
        """
        self = self * coeff
        coeff: float
        """
        self.values *= coeff

    def assign_madd(self, update, coeff):
        """
        self = self + update * coeff
        coeff: float
        """
        self.positions = np.concatenate((self.positions, update.positions))
        self.values = np.concatenate((self.values, update.values * coeff))

###############################################################################
#                                                                             #
#                                  MODEL                                      #
#                                                                             #
###############################################################################


Features = Update


class LinearModel:
    """
    A thing that computes score and gradient for given features.
    """

    def __init__(self, n):
        self._params = Value(n)

    def params(self):
        return self._params

    def score(self, features):
        """
        features: Update
        """
        return self._params.dot(features)

    def gradient(self, features, score):
        return features


###############################################################################
#                                                                             #
#                                    HYPO                                     #
#                                                                             #
###############################################################################


Hypo = namedtuple('Hypo', ['prev', 'pos', 'tagged_word', 'score'])
# prev: previous Hypo
# pos: position of word (0-based)
# tagged_word: tagging of source_sentence[pos]
# score: sum of scores over edges

###############################################################################
#                                                                             #
#                              FEATURE COMPUTER                               #
#                                                                             #
###############################################################################


def h(x):
    """
    Compute CityHash of any object.
    Can be used to construct features.
    """
    return cityhash.CityHash64(repr(x))


TaggerParams = namedtuple('FeatureParams', [
    'src_window',
    'dst_order',
    'max_suffix',
    'beam_size',
    'nparams'
    ])


class FeatureComputer:
    def __init__(self, tagger_params, source_sentence):
        self._tagger_params = tagger_params
        self._source_sentence = source_sentence

    def compute_features(self, hypo):
        """
        Compute features for a given Hypo and return Update.
        """
        word = hypo.tagged_word.text
        tag = hypo.tagged_word.tag

        names = ['has numbers', 'has capitalized', 'has hyphen']
        f = [(re.search(pattern, word) is None, name)
             for pattern, name in zip(['[0-9]', '[A-Z]', '-'], names)]

        max_subsection = self._tagger_params.max_suffix + 1
        f += [(word[:i], str(i + 1) + ' suffix')
              for i in range(1, max_subsection)]
        f += [(word[-i:], str(i + 1) + ' prefix')
              for i in range(1, max_subsection)]

        max_depht = self._tagger_params.dst_order - 1
        tags = tuple()
        for i in range(1, max_depht):
            if hypo.prev is None:
                tags += (None,)
            else:
                hypo = hypo.prev
                tags += (hypo.tagged_word.tag,)
            f.append((tags, str(i) + ' previous tags'))

        def get_elem_or_None(sent, i):
            if i < 0 or i >= len(sent):
                return None
            else:
                return sent[i]

        def get_name(i):
            if i < 0:
                return str(-i) + ' word back'
            elif i > 0:
                return str(i) + ' word next'
            else:
                return 'word'

        f += [(get_elem_or_None(self._source_sentence, hypo.pos + i),
               get_name(i))
              for i in range(-self._tagger_params.src_window,
                             self._tagger_params.src_window + 1)]

        hash_size = self._tagger_params.nparams
        f = [h((tag,) + feature) % hash_size for feature in f]
        return Update(positions=f, values=np.ones(len(f)))


###############################################################################
#                                                                             #
#                                BEAM SEARCH                                  #
#                                                                             #
###############################################################################


class BeamSearchTask:
    """
    An abstract beam search task. Can be used with beam_search() generic
    function.
    """

    def __init__(self, tagger_params, source_sentence, model, tags):
        self._tagger_params = tagger_params
        self._source_sentence = source_sentence
        self._model = model
        self._tags = tags
        self._feature_computer = FeatureComputer(tagger_params, source_sentence)

    def total_num_steps(self):
        """
        Number of hypotheses between beginning and end (number of words in
        the sentence).
        """
        return len(self._source_sentence)

    def beam_size(self):
        return self._tagger_params.beam_size

    def _get_next_hypo(self, hypo, tag):
        pos = 0 if hypo is None else hypo.pos + 1
        tagged_word = TaggedWord(self._source_sentence[pos], tag)
        new_hypo = Hypo(hypo, pos, tagged_word, 0)
        features = self._feature_computer.compute_features(new_hypo)
        score = self._model.score(features)
        if hypo is not None:
            score += hypo.score
        return new_hypo._replace(score=score)

    def expand(self, hypo):
        """
        Given Hypo, return a list of its possible expansions.
        'hypo' might be None -- return a list of initial hypos then.

        Compute hypotheses' scores inside this function!
        """
        return [self._get_next_hypo(hypo, tag) for tag in self._tags]

    def recombo_hash(self, hypo):
        """
        If two hypos have the same recombination hashes, they can be collapsed
        together, leaving only the hypothesis with a better score.
        """
        # this implementation returns typle of tags, which accounts in features
        result = tuple()
        for i in range(self._tagger_params.dst_order):
            result += (hypo.tagged_word.tag,)
            hypo = hypo.prev
        return result


def sort_and_cut(hypos_list, remain_num):
    return sorted(hypos_list, key=lambda x: -x.score)[:remain_num]


def beam_search(beam_search_task):
    """
    Return list of stacks.
    Each stack contains several hypos, sorted by score in descending
    order (i.e. better hypos first).
    """
    if beam_search_task.total_num_steps() == 0:
        return [[]]

    beam_size = beam_search_task.beam_size()
    first_stack = beam_search_task.expand(None)
    stacks = [sort_and_cut(first_stack, beam_size)]

    for _ in range(1, beam_search_task.total_num_steps()):
        stack = []
        for hypo in stacks[-1]:
            stack += beam_search_task.expand(hypo)

        stacks.append(sort_and_cut(stack, beam_size))

    return stacks


###############################################################################
#                                                                             #
#                            OPTIMIZATION TASKS                               #
#                                                                             #
###############################################################################


class OptimizationTask:
    """
    Optimization task that can be used with sgd().
    """

    def params(self):
        """
        Parameters which are optimized in this optimization task.
        Return Value.
        """
        raise NotImplementedError()

    def loss_and_gradient(self, golden_sentence):
        """
        Return (loss, gradient) on a specific example.

        loss: float
        gradient: Update
        """
        raise NotImplementedError()


class UnstructuredPerceptronOptimizationTask(OptimizationTask):
    def __init__(self, tagger_params, tags):
        raise NotImplementedError()

    def params(self):
        raise NotImplementedError()

    def loss_and_gradient(self, golden_sentence):
        raise NotImplementedError()


class StructuredPerceptronOptimizationTask(OptimizationTask):
    def __init__(self, tagger_params, tags):
        self.tagger_params = tagger_params
        self.model = LinearModel(tagger_params.nparams)
        self.tags = tags

    def params(self):
        return self.model.params()

    def loss_and_gradient(self, golden_sentence):
        # Do beam search.
        beam_search_task = BeamSearchTask(
            self.tagger_params,
            [golden_tagged_word.text for golden_tagged_word in golden_sentence],
            self.model,
            self.tags
            )
        stacks = beam_search(beam_search_task)

        # Compute chain of golden hypos (and their scores!).
        golden_hypo = None
        golden_source_sentence = [word.text for word in golden_sentence]
        feature_computer = FeatureComputer(self.tagger_params,
                                           golden_source_sentence)
        max_violation = 0

        for i, word in enumerate(golden_sentence):
            new_golden_hypo = Hypo(golden_hypo, i, word, 0)
            feature = feature_computer.compute_features(new_golden_hypo)
            score = self.model.score(feature)
            if golden_hypo is not None:
                score += golden_hypo.score

            golden_hypo = new_golden_hypo._replace(score=score)

            violation = stacks[i][-1].score - score
            if violation > max_violation:
                max_violation = violation
                rival_head = stacks[i][0]
                golden_head = golden_hypo

        # Find where to update.
        if max_violation == 0:
            rival_head = stacks[-1][0]
            golden_head = golden_hypo

        # Compute gradient.
        grad = Update()
        while golden_head and rival_head:
            for head, multiplyer in zip([rival_head, golden_head], [1, -1]):
                features = feature_computer.compute_features(head)
                add_grad = self.model.gradient(features, score=None)
                grad.assign_madd(add_grad, multiplyer)

            golden_head = golden_head.prev
            rival_head = rival_head.prev

        return grad


###############################################################################
#                                                                             #
#                                    SGD                                      #
#                                                                             #
###############################################################################


SGDParams = namedtuple('SGDParams', [
    'epochs',
    'learning_rate',
    'minibatch_size',
    'average'  # int
    ])


def make_batches(dataset, minibatch_size):
    """
    Make list of batches from a list of examples.
    """
    sequence = list(range(ceil(len(dataset) / minibatch_size)))
    shuffle(sequence)
    for i in sequence:
        yield dataset[i*minibatch_size:(i + 1)*minibatch_size]


def sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn):
    """
    Run (averaged) SGD on a generic optimization task. Modify optimization
    task's parameters.

    After each epoch (and also before and after the whole training),
    run after_each_epoch_fn().
    """
    after_each_epoch_fn()
    average = sgd_params.average

    if average:
        params_sum = Value(len(optimization_task.params()))
        added_cnt = 0

    for _ in range(sgd_params.epochs):
        batches = make_batches(dataset, sgd_params.minibatch_size)
        for ind, batch in enumerate(batches):
            grad = Update()
            for sent in batch:
                sent_grad = optimization_task.loss_and_gradient(sent)
                grad.assign_madd(sent_grad, 1)
            optimization_task.params().assign_madd(grad,
                                                   -sgd_params.learning_rate)
            if average and ind % average == average - 1:
                params_sum.assign_madd(optimization_task.params(), 1)
                added_cnt += 1
        after_each_epoch_fn()

    if average:
        params_sum.assign_mul(1 / added_cnt)
        optimization_task.params().assign(params_sum)
        after_each_epoch_fn()


###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


# - Train - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def TRAIN_add_cmdargs(subp):
    p = subp.add_parser('train')

    p.add_argument('--tags',
                   help='tags file', type=str, default='data/tags')
    p.add_argument('--dataset',
                   help='train dataset', default='data/en-ud-train.conllu')
    p.add_argument('--dataset-dev',
                   help='dev dataset', default='data/en-ud-dev.conllu')
    p.add_argument('--model',
                   help='NPZ model', type=str, default='model.npz')
    p.add_argument('--sgd-epochs',
                   help='SGD number of epochs', type=int, default=15)
    p.add_argument('--sgd-learning-rate',
                   help='SGD learning rate', type=float, default=0.01)
    p.add_argument('--sgd-minibatch-size',
                   help='SGD minibatch size (in sentences)', type=int,
                   default=32)
    p.add_argument('--sgd-average',
                   help='SGD average every N batches', type=int, default=32)
    p.add_argument('--tagger-src-window',
                   help='Number of context words in input sentence to use' +
                   'for features',
                   type=int, default=2)
    p.add_argument('--tagger-dst-order',
                   help='Number of context tags in output tagging to use' +
                   'for features',
                   type=int, default=3)
    p.add_argument('--tagger-max-suffix',
                   help='Maximal number of prefix/suffix letters to use' +
                   'for features',
                   type=int, default=4)
    p.add_argument('--beam-size',
                   help='Beam size (0 means unstructured)', type=int,
                   default=4)
    p.add_argument('--nparams',
                   help='Parameter vector size', type=int, default=2**22)

    return 'train'


def TRAIN(cmdargs):
    # Beam size.
    optimization_task_cls = StructuredPerceptronOptimizationTask
    if cmdargs.beam_size == 0:
        cmdargs.beam_size = 1
        optimization_task_cls = UnstructuredPerceptronOptimizationTask

    # Parse cmdargs.
    tags = read_tags(cmdargs.tags)
    dataset = read_tagged_sentences(cmdargs.dataset)
    dataset_dev = read_tagged_sentences(cmdargs.dataset_dev)
    params = None
    if os.path.exists(cmdargs.model):
        params = pickle.load(open(cmdargs.model, 'rb'))
    sgd_params = SGDParams(
        epochs=cmdargs.sgd_epochs,
        learning_rate=cmdargs.sgd_learning_rate,
        minibatch_size=cmdargs.sgd_minibatch_size,
        average=cmdargs.sgd_average
        )
    tagger_params = TaggerParams(
        src_window=cmdargs.tagger_src_window,
        dst_order=cmdargs.tagger_dst_order,
        max_suffix=cmdargs.tagger_max_suffix,
        beam_size=cmdargs.beam_size,
        nparams=cmdargs.nparams
        )

    # Load optimization task
    optimization_task = optimization_task_cls(tagger_params, tags)
    if params is not None:
        print('\n\nLoading parameters from %s\n\n' % cmdargs.model)
        optimization_task.params().assign(params)

    # Validation.
    def after_each_epoch_fn():
        model = LinearModel(cmdargs.nparams)
        model.params().assign(optimization_task.params())
        # lol = model.params().values
        # print(lol[lol != 1])
        tagged_sentences = tag_sentences(dataset_dev, tagger_params, model,
                                         tags)
        q = pprint.pformat(tagging_quality(out=tagged_sentences,
                                           ref=dataset_dev))
        print()
        print(q)
        print()

        # Save parameters.
        print('\n\nSaving parameters to %s\n\n' % cmdargs.model)
        pickle.dump(optimization_task.params(), open(cmdargs.model, 'wb'))

    # Run SGD.
    sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn)


# - Test  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def TEST_add_cmdargs(subp):
    p = subp.add_parser('test')

    p.add_argument('--tags',
                   help='tags file', type=str, default='data/tags')
    p.add_argument('--dataset',
                   help='test dataset', default='data/en-ud-dev.conllu')
    p.add_argument('--model',
                   help='NPZ model', type=str, default='model.npz')
    p.add_argument('--tagger-src-window',
                   help='Number of context words in input sentence to use' +
                   'for features',
                   type=int, default=2)
    p.add_argument('--tagger-dst-order',
                   help='Number of context tags in output tagging to use' +
                   'for features',
                   type=int, default=3)
    p.add_argument('--tagger-max-suffix',
                   help='Maximal number of prefix/suffix letters to use' +
                   'for features',
                   type=int, default=4)
    p.add_argument('--beam-size',
                   help='Beam size', type=int, default=4)

    return 'test'


def tag_sentences(dataset, tagger_params, model, tags):
    """
    Tag all sentences in dataset. Dataset is a list of TaggedSentence; while
    tagging, ignore existing tags.
    """
    def get_tag_sent(sent):
        if len(sent) == 0:
            return []
        sent = [word.text for word in sent]
        beam_search_task = BeamSearchTask(tagger_params, sent, model, tags)
        stacks = beam_search(beam_search_task)
        hypo = stacks[-1][0]
        result = []
        for word in reversed(sent):
            result.append(TaggedWord(word, hypo.tagged_word.tag))
            hypo = hypo.prev
        return list(reversed(result))

    return [get_tag_sent(sent) for sent in dataset]


def TEST(cmdargs):
    # Parse cmdargs.
    tags = read_tags(cmdargs.tags)
    dataset = read_tagged_sentences(cmdargs.dataset)
    params = pickle.load(open(cmdargs.model, 'rb'))
    tagger_params = TaggerParams(
        src_window=cmdargs.tagger_src_window,
        dst_order=cmdargs.tagger_dst_order,
        max_suffix=cmdargs.tagger_max_suffix,
        beam_size=cmdargs.beam_size,
        nparams=len(params),
        )

    # Load model.
    model = LinearModel(params.values.shape[0])
    model.params().assign(params)

    # Tag all sentences.
    tagged_sentences = tag_sentences(dataset, tagger_params, model, tags)

    # Write tagged sentences.
    for tagged_sentence in tagged_sentences:
        write_tagged_sentence(tagged_sentence, sys.stdout)

    # Measure and print quality.
    q = pprint.pformat(tagging_quality(out=tagged_sentences, ref=dataset))
    print(q, file=sys.stderr)


# - Main  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def main():
    # Create parser.
    p = argparse.ArgumentParser('tagger.py')
    subp = p.add_subparsers(dest='cmd')

    # Add subcommands.
    train = TRAIN_add_cmdargs(subp)
    test = TEST_add_cmdargs(subp)

    # Parse.
    cmdargs = p.parse_args()

    # Run.
    if cmdargs.cmd == train:
        TRAIN(cmdargs)
    elif cmdargs.cmd == test:
        TEST(cmdargs)
    else:
        p.error('No command')


if __name__ == '__main__':
    main()
