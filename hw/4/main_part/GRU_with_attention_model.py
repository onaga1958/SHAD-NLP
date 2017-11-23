import tensorflow as tf
import keras.layers as L
import numpy as np

from basic_model import infer_length


class AttentionGRUTranslationModel:
    _BILINEAR = 'bilinear'
    _PERCEPTRON = 'perceptron'
    _DOT_PRODUCT = 'dot product'

    def __init__(self, name, inp_voc, out_voc, emb_size, hid_size,
                 attention_function, bidirectional=True, attention_size=None):

        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.bidirectional = bidirectional
        self.attention_function = attention_function
        self.hid_size = hid_size

        with tf.variable_scope(name):
            if attention_function == self._BILINEAR:
                self.attention_W = tf.Variable(np.zeros((hid_size, hid_size)), dtype=np.float32)
            elif attention_function == self._PERCEPTRON:
                if attention_size is None:
                    attention_size = hid_size
                self.attention_W_hid = tf.Variable(np.zeros((attention_size, hid_size)),
                                                   dtype=np.float32)
                self.attention_W_emb = tf.Variable(np.zeros((attention_size, hid_size)),
                                                   dtype=np.float32)
                self.attention_v = tf.Variable(np.zeros(attention_size), dtype=np.float32)
            elif attention_function == self._DOT_PRODUCT:
                pass
            else:
                raise ValueError('Unknown attention_function: %s' % attention_function)
            self.emb_inp = L.Embedding(len(inp_voc), emb_size)
            self.emb_out = L.Embedding(len(out_voc), emb_size)
            if bidirectional:
                self.enc0 = tf.nn.rnn_cell.GRUCell(hid_size // 2)
                self.enc1 = tf.nn.rnn_cell.GRUCell(hid_size // 2)
            else:
                self.enc0 = tf.nn.rnn_cell.GRUCell(hid_size)
            self.dec0 = tf.nn.rnn_cell.GRUCell(hid_size)
            self.logits = L.Dense(len(out_voc))

            # run on dummy output to .build all layers (and therefore create weights)
            inp = tf.placeholder('int32', [None, None])
            out = tf.placeholder('int32', [None, None])
            encodings, h0 = self.encode(inp)
            attention = self.attention(encodings, h0)
            h1 = self.decode(h0, out[:, 0], attention)
            # h2 = self.decode(h1,out[:,1]) etc.

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        inp_emb = self.emb_inp(inp)
        if self.bidirectional:
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(self.enc0, self.enc1, inp_emb,
                                                                   sequence_length=inp_lengths,
                                                                   dtype=inp_emb.dtype)
            outputs, final_state = tf.concat(outputs, 2), tf.concat(final_state, 1)
        else:
            outputs, final_state = tf.nn.dynamic_rnn(self.enc0, inp_emb,
                                                     sequence_length=inp_lengths,
                                                     dtype=inp_emb.dtype)

        return outputs, final_state

    def decode(self, prev_state, prev_tokens, attention, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """

        prev_emb = self.emb_out(prev_tokens[:, None])[:, 0]

        dec_input = tf.concat((prev_emb, attention), axis=1)
        new_dec_out, new_dec_state = self.dec0(dec_input, prev_state)

        output_logits = self.logits(new_dec_out)

        return [new_dec_state], output_logits

    def attention(self, encodings, prev_hidden):
        if self.attention_function == self._BILINEAR:
            scores = tf.tensordot(encodings, self.attention_W, axes=[[2], [0]])
            scores = tf.transpose(scores, [1, 0, 2])
            scores = tf.reduce_sum(scores * prev_hidden, axis=2)

        if self.attention_function == self._PERCEPTRON:
            emb_multiply_result = tf.tensordot(encodings, self.attention_W_emb,
                                               axes=[[2], [1]])

            # (batch_size, seq_len, attention_size) -> (seq_len, batch_size, attention_size)
            emb_multiply_result = tf.transpose(emb_multiply_result, [1, 0, 2])
            hid_multiply_result = tf.tensordot(prev_hidden, self.attention_W_hid,
                                               axes=[[1], [1]])
            scores = emb_multiply_result + hid_multiply_result
            scores = tf.tanh(scores)
            scores = tf.tensordot(scores, self.attention_v, axes=[[2], [0]])

        if self.attention_function == self._DOT_PRODUCT:
            transposed_encodings = tf.transpose(encodings, [1, 0, 2])
            scores = tf.reduce_sum(transposed_encodings * prev_hidden, axis=2)

        probs = tf.nn.softmax(scores, dim=0)
        attention = tf.reduce_sum(probs * tf.transpose(encodings, [2, 1, 0]), axis=1)
        attention = tf.transpose(attention, [1, 0])

        return attention  # , probs

    def symbolic_score(self, inp, out, eps=1e-30, **flags):
        """
        Takes symbolic int32 matrices of hebrew words and their english translations.
        Computes the log-probabilities of all possible english characters given english prefices and hebrew word.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param out: output sequence, int32 matrix of shape [batch,time]
        :return: log-probabilities of all possible english characters of shape [bath,time,n_tokens]

        NOTE: log-probabilities time axis  is synchronized with out
        In other words, logp are probabilities of __current__ output at each tick, not the next one
        therefore you can get likelihood as logprobas * tf.one_hot(out,n_tokens)
        """
        encodings, encode_sent = self.encode(inp, **flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_ix)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)

        def step(blob, y_prev):
            h_prev = blob[0]
            attention = self.attention(encodings, h_prev)
            h_new, logits = self.decode(h_prev, y_prev, attention, **flags)
            return list(h_new) + [logits]

        results = tf.scan(step, initializer=[encode_sent]+[first_logits],
                          elems=tf.transpose(out))

        # gather state and logits, each of shape [time,batch,...]
        logits_seq = results[-1]

        # add initial state and logits
        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)

        # convert from [time,batch,...] to [batch,time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])

        return tf.nn.log_softmax(logits_seq)

    def symbolic_translate(self, inp, greedy=False, max_len=None, eps=1e-30, **flags):
        """
        takes symbolic int32 matrix of hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param greedy: if greedy, takes token with highest probablity at each tick.
            Otherwise samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :return: output tokens int32[batch,time] and
                 log-probabilities of all tokens at each tick, [batch,time,n_tokens]
        """
        encodings, encode_sent = self.encode(inp, **flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_ix)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)
        max_len = tf.reduce_max(tf.shape(inp)[1])*2

        def step(blob, t):
            h_prev, y_prev = blob[0], blob[-1]
            attention = self.attention(encodings, h_prev)
            h_new, logits = self.decode(h_prev, y_prev, attention, **flags)
            y_new = tf.argmax(logits, axis=-1) if greedy else tf.multinomial(logits, 1)[:, 0]
            return list(h_new) + [logits, tf.cast(y_new, y_prev.dtype)]

        results = tf.scan(step, initializer=[encode_sent] + [first_logits, bos],
                          elems=tf.range(max_len))

        # gather state, logits and outs, each of shape [time,batch,...]
        logits_seq, out_seq = results[-2], results[-1]

        # add initial state, logits and out
        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)
        out_seq = tf.concat((bos[None], out_seq), axis=0)

        # convert from [time,batch,...] to [batch,time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        out_seq = tf.transpose(out_seq)

        return out_seq, tf.nn.log_softmax(logits_seq)
