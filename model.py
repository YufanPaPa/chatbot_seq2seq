import time
import sys
import numpy as np
from module import attention_decoder, mlp, embedding, multihead_attention, feedforward, ptr_net, summ, pointer, \
    dot_attention, attention_decoder_v2
import tensorflow as tf

EPS = 1e-8
INF = 1e30


class CommentModel(object):
    def __init__(self, args, iterator):
        self.args = args
        # self.vocab_size = args.vocab_size
        self.build_graph(iterator)

    def input_batch_data(self, iterator):

        user_indice, query, response, target = iterator.get_next()
        self.target = target

        self.user_indice = user_indice
        self.query = query
        self.response = response

        # specify the shape
        self.user_indice.set_shape((self.args.batch_size,1))
        self.response.set_shape((self.args.batch_size, self.args.max_response_steps))
        self.query.set_shape((self.args.batch_size, self.args.max_query_len))


    def _add_test_placeholder(self):  # add the placeholder for test.
        self.user_indice = tf.placeholder(tf.int32, [self.args.batch_size, 1])
        self.query = tf.placeholder(tf.int32, [self.args.batch_size, self.args.max_query_len_test])
        self.response = tf.placeholder(tf.int32, [self.args.batch_size, 1])

        # self.span_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.max_span_seq_len_test])
        # self.span_seq_len = tf.placeholder(tf.int32, [self.args.batch_size])
        # self.x = tf.placeholder(tf.int32, [self.args.batch_size, self.args.max_span_num])
        # self.span_num = tf.placeholder(tf.int32, [self.args.batch_size])

    def _add_len_mask(self):

        self.query_mask = tf.cast(tf.cast(self.query, tf.bool), tf.float32)
        self.query_len = tf.reduce_sum(tf.cast(self.query_mask, tf.int32), axis=1)
        self.response_mask = tf.cast(tf.cast(self.response, tf.bool), tf.float32)
        self.response_len = tf.reduce_sum(tf.cast(self.response_mask, tf.int32), axis=1)

        # self.general_special_bool_mask = tf.cast(tf.cast(self.general_special_bool, tf.bool), tf.float32)

        # ss_len = self.span_seq.get_shape()[1].value
        # self.span_seq_mask = tf.cast(tf.sequence_mask(self.span_seq_len, ss_len), tf.float32)

        if self.args.mode in ['train', 'eval', 'debug']:
            if self.args.pointer_gen:
                self.target_extend_vocab_mask = tf.cast(tf.cast(self.target_extend_vocab, tf.bool), tf.float32)
                self.target_extend_vocab_len = tf.reduce_sum(tf.cast(self.target_extend_vocab_mask, tf.int32))
            else:
                self.target_mask = tf.cast(tf.cast(self.target, tf.bool), tf.float32)
                self.target_len = tf.reduce_sum(tf.cast(self.target_mask, tf.int32), axis=1)
            # self.span_mask = tf.cast(tf.sequence_mask(self.span_num, self.args.max_span_num), tf.float32)
        # self.general_special_bool_mask = -INF * (1 - self.target_mask)
        # self.general_special_bool_len = self.target_len

    def _add_embedding(self, scope_name='embedding'):
        with tf.variable_scope(scope_name):
            args = self.args
            self.user_indice = tf.squeeze(self.user_indice)
            user_indice_emb = embedding(self.user_indice, 50000, args.token_emb_dim, scope='user_id_emb', reuse=tf.AUTO_REUSE)
            query_emb = embedding(self.query, args.vocab_size_encoder, args.token_emb_dim, scope='encoder_emb',
                                  reuse=tf.AUTO_REUSE)
            emb_dec_inputs = [
                embedding(_, args.vocab_size_decoder, args.token_emb_dim, scope='decoder_emb', reuse=tf.AUTO_REUSE)
                for _ in tf.unstack(self.response, axis=1)]


            # emb_dec_inputs : length = max_dec_len, each element:(batch_size,token_emb_dim)
            # general_special_emb : length = max_dec_len, each element:(batch_size,pos_emb_dim)
        return user_indice_emb, query_emb, emb_dec_inputs

    def _add_joint_net(self):
        with tf.variable_scope('joint_network'):
            args = self.args
            dropout_keep_prob = args.dropout_keep_prob
            hidden_dim = self.args.hidden_dim
            vsize = self.args.vocab_size_decoder

            self.rand_unif_init = tf.random_uniform_initializer(-args.rand_unif_init_mag, args.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=args.trunc_norm_init_std)

            # add embedding
            self.user_indice_emb, self.query_emb, self.dec_inputs_emb = self._add_embedding()
            self.log_infos = []

            # encoder
            with tf.variable_scope('query_encode'):
                fw_cell = tf.nn.rnn_cell.GRUCell(hidden_dim / 2)
                bw_cell = tf.nn.rnn_cell.GRUCell(hidden_dim / 2)
                outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    self.query_emb,
                    sequence_length=self.query_len,
                    dtype=tf.float32
                )
                self.query_enc = tf.concat(outputs, 2)  # [B, max_query_len, hidden_dim]

            with tf.variable_scope('generation'):
                cell = tf.nn.rnn_cell.GRUCell(hidden_dim, kernel_initializer=self.rand_unif_init,
                                              bias_initializer=self.rand_unif_init)
                dec_in_state = summ(self.query_enc, hidden_dim,
                                         self.query_mask, dropout_keep_prob,
                                         self.is_train)
                self.dec_in_state = mlp(tf.concat([dec_in_state,self.user_indice_emb],1),2,hidden_dim,scope="merge_info")

                self.dec_outputs, self.dec_output_state = attention_decoder_v2(
                    self.dec_inputs_emb, self.dec_in_state,
                    self.query_enc,
                    self.user_indice_emb, self.query_mask,
                    cell)


                self.check_dec_outputs = tf.check_numerics(tf.stack(self.dec_outputs, axis=1),
                                                            "dec_outputs is wrong!!")

                # add projection
                with tf.variable_scope('output_projection'):
                    vocab_scores = []  # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
                    for i, output in enumerate(self.dec_outputs):
                        if i > 0:
                            tf.get_variable_scope().reuse_variables()
                        vocab_score = tf.layers.dense(output, vsize, kernel_initializer=self.trunc_norm_init,
                                                      bias_initializer=self.trunc_norm_init,
                                                      name='vocab_proj')


                        # self.log_infos.append(self.general_special_bool_mask)
                        # self.log_infos.append(tf.sigmoid(self.outputs_special_general))
                        # self.log_infos.append(self.target_general_special_bool)
                        # vocab_score = vocab_score + self.vocab_score_mask
                        # vocab_scores.append(tf.layers.dense(output, vsize, kernel_initializer=self.trunc_norm_init,
                        # bias_initializer=self.trunc_norm_init,
                        # name='vocab_proj'))
                        vocab_scores.append(vocab_score)
                        self.log_infos.append(vocab_score)

                    # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                    # The words are in the order they appear in the vocabulary file.
                    vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]

                self.final_dists = vocab_dists
                #self.check_vocab_scores = tf.check_numerics(tf.stack(vocab_scores, axis=1),
                                                              #"vocab_scores is wrong!!")

            # add loss
            if args.mode in ['train', 'eval']:
                with tf.variable_scope('loss'):
                    # generation loss
                    # vocab_scores:batch_size*max_seq_len*vsize
                    # target:batch_size*max_seq_len
                    # target_general_special_bool = tf.cast(self.target_general_special_bool, tf.float32)
                    # self.masked_target_general_special_bool = target_general_special_bool * self.target_mask

                    self.loss_gen = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1),
                                                                     self.target, self.target_mask)
                    self.ppl = tf.exp(self.loss_gen)
                    # classification loss
                    # general_special:batch_size*max_seq_len
                    # outputs_special_general:batch_size*max_seq_len
                    # self.loss_classification = tf.losses.sigmoid_cross_entropy(self.masked_target_general_special_bool,
                    # self.outputs_special_general,
                    # reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
                    # self.loss_classification = tf.losses.log_loss(self.masked_target_general_special_bool,self.outputs_special_general_sigmoid,
                    # reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)


                    tf.summary.scalar('gen_loss', self.loss_gen)
                    tf.summary.scalar('ppl', self.ppl)

                    # total_loss
                    self.loss = self.loss_gen
                    # self.loss = self.loss_gen
                    if args.mode == 'train':
                        self._add_train_op(self.loss)
                        # self._add_train_op(self.loss)

            if args.mode == "decode":
                # We run decode beam search mode one decoder step at a time
                # final_dists is a singleton list containing shape (batch_size, extended_vsize)
                #target_general_special_bool = tf.transpose(self.target_general_special_bool)
                #target_general_special_bool_t = tf.one_hot(target_general_special_bool[0], 2)
                #general_special_mask = tf.stack([self.general_mask, self.special_mask], axis=0)
                #self.vocab_score_mask = tf.matmul(target_general_special_bool_t, general_special_mask)

                assert len(self.final_dists) == 1
                final_dists = self.final_dists[0]
                #final_dists = final_dists * self.vocab_score_mask
                topk_probs, self._topk_ids = tf.nn.top_k(final_dists, args.batch_size * 5)
                # take the k largest probs. note batch_size=beam_size in decode mode
                self._topk_log_probs = tf.log(topk_probs)

    def _add_train_op(self, loss_to_minimize):
        args = self.args
        # Apply optimizer
        self.lr = tf.get_variable('lr', dtype=tf.float32, initializer=tf.constant(args.lr, dtype=tf.float32),
                                  trainable=False)
        tf.summary.scalar('lr', self.lr)
        if args.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.lr, initial_accumulator_value=args.adagrad_init_acc)
        elif args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif args.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif args.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        else:
            raise NameError(f'args optimizer {args.optimizer} invalid.')

        grads = optimizer.compute_gradients(loss_to_minimize)
        gradients, vars = zip(*grads)
        capped_grads, global_norm = tf.clip_by_global_norm(gradients, args.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)
        self._train_op = optimizer.apply_gradients(zip(capped_grads, vars), global_step=self.global_step)

    def build_graph(self, iterator=None):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        print(f'{time.asctime()} - Build model..')
        t0 = time.time()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_train = tf.Variable(True, name='is_train', trainable=False)
        if self.args.mode in ['train', 'eval', 'debug']:
            self.input_batch_data(iterator)
        if self.args.mode == 'decode':
            self._add_test_placeholder()
        self._add_len_mask()
        self._add_joint_net()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        print(f'{time.asctime()} - Time to build graph: {t1-t0} seconds')

    def run_encoder(self, sess, batch, pointer_gen=False):
        user_indice = np.squeeze(batch.user_indice)
        feed_dict = {
            self.user_indice: user_indice,
            self.query: batch.query_ids,
        }
        (user_indice_emb, query_enc, query_mask, dec_in_state) = sess.run(
            [self.user_indice_emb, self.query_enc, self.query_mask, self.dec_in_state],
            feed_dict
        )
        return user_indice_emb, query_enc, query_mask, dec_in_state[0]

    def decode_onestep(self, sess, batch, latest_tokens, query_enc, query_mask, user_indice_emb, dec_init_states):

        feed_dict = {
            self.user_indice_emb: user_indice_emb,
            self.query_enc: query_enc,
            self.query_mask: query_mask,
            self.dec_in_state: dec_init_states,
            self.response: np.transpose(np.array([latest_tokens]))
        }


        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "all_probs": self.final_dists,
            'states': self.dec_output_state
        }
        results = sess.run(to_return, feed_dict=feed_dict)

        # return _topk_ids, _topk_log_probs, results['states']
        return results['ids'], results['probs'], results['states']


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """
    padding_mask = tf.cast(padding_mask, tf.float32)
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average


def _mask_softmax(logits, padding_mask):
    dist = tf.nn.softmax(logits, axis=1)
    dist *= padding_mask  # apply mask
    masked_sums = tf.reduce_sum(dist, axis=1)  # shape (batch_size)
    return dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize


def add_epsilon(dist, epsilon=sys.float_info.epsilon):
    epsilon_mask = tf.ones_like(dist) * epsilon
    return dist + epsilon_mask
