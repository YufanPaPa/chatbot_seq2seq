# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This file contains code to run beam search decoding"""
INF = 1e30

import vocabulary
import copy
import time
import numpy as np


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, source_words, log_probs, state, attn_dists, p_gens):
        """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
    """
        self.tokens = tokens
        self.source_words = source_words
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens

    def extend(self, token, word, log_prob, state, attn_dist, p_gen):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
        return Hypothesis(
            tokens=self.tokens + [token],
            source_words=self.source_words + [word],
            log_probs=self.log_probs + [log_prob],
            state=state,
            attn_dists=self.attn_dists + [attn_dist],
            p_gens=self.p_gens + [p_gen])

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)  # every hyp pad [START] with 0.0 prob at first.

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        # return self.log_prob / max(len(self.tokens) - 1, 1)
        return self.log_prob


def make_one_hot(args, data):
    return (np.arange(args.vocab_size_decoder) == data[:, None]).astype(np.integer)


def run_beam_search(args, sess, model, vocab, batch):
    """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
    # Run the encoder to get the encoder hidden states and decoder initial state


    user_indice_emb, query_enc, query_mask, dec_in_state = model.run_encoder(sess, batch)

    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(vocabulary.START_DECODING)],
                       source_words=[vocabulary.START_DECODING],
                       log_probs=[0.0],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[]  # zero vector of length attention_length
                       ) for _ in range(args.beam_size)]

    steps = 0

    groups = 3
    group_size = int(args.beam_size / groups)

    results = [[] for _ in range(groups)]

    while steps < args.max_response_steps and len(results) < args.beam_size:
        # print(steps)
        all_words = []
        latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
        latest_tokens = [t if t < vocab.size() else vocab.word2id(vocabulary.UNKNOWN_TOKEN) for t in
                         latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
        states = [h.state for h in hyps]  # list of current decoder states of the hypotheses

        # Run one step of the decoder to get the new info
        topk_ids, topk_log_probs, new_states = model.decode_onestep(
            sess=sess,
            batch=batch,
            latest_tokens=latest_tokens,
            query_enc=query_enc,
            query_mask=query_mask,
            user_indice_emb=user_indice_emb,
            dec_init_states=states
        )
        # print("topk_ids: "+ str(topk_ids))
        # print("topk_log_probs: "+str(topk_log_probs))

        # Extend each hypothesis and collect them all in all_hyps
        # num_orig_hyps = 1 if steps == 0 else len(hyps)
        num_orig_hyps = 1 if steps == 0 else group_size
        # print(str(steps)+"\n")
        # print(topk_log_probs)
        # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
        new_hyps = []
        for group in range(groups):
            cur_group_hyps = []
            for idx in range(num_orig_hyps):
                i = group * group_size + idx
                h, new_state, attn_dist, p_gen = hyps[i], new_states[
                    i], None, None
                # take the ith hypothesis and new decoder state info
                for j in range(group_size * 5):  # for each of the top 2*beam_size hyps:
                    # Extend the ith hypothesis with the jth option

                    if vocab.id2word(topk_ids[i, j]) == vocabulary.UNKNOWN_TOKEN:
                        continue

                    topk_log_probs[i, j] -= j * np.log(2)

                    if group != 0 and topk_ids[i, j] in all_words:
                        topk_log_probs[i, j] = topk_log_probs[i, j] - np.log(5)

                    if vocab.id2word(topk_ids[i, j]) in ['〈', 'あなた', 'ちゃん', '〉','え 、']:
                        continue

                    if vocab.id2word(topk_ids[i, j]) in h.source_words:
                        # topk_log_probs[i, j] = topk_log_probs[i, j] - np.log(2)
                        continue

                    new_hyp = h.extend(token=topk_ids[i, j],
                                       word=vocab.id2word(topk_ids[i, j]),
                                       log_prob=topk_log_probs[i, j],
                                       state=new_state,
                                       attn_dist=attn_dist,
                                       p_gen=p_gen
                                       )
                    cur_group_hyps.append(new_hyp)

            # Filter and collect any hypotheses that have produced the end token.
            sorted_hyps = sort_hyps(cur_group_hyps)
            for h in sorted_hyps:  # in order of most likely h
                if h.latest_token == vocab.word2id(vocabulary.STOP_DECODING):  # if stop token is reached...
                    # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                    # print(h.latest_token)
                    if steps >= args.min_dec_steps and len(results[group]) < group_size:
                        results[group].append(h)
                else:  # hasn't reached stop token, so continue to extend this hypothesis
                    all_words.extend(h.tokens)
                    new_hyps.append(h)

                if len(new_hyps) == group_size * (group + 1):
                    # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                    break

        steps += 1

        if len(hyps) == 0:
            break
        while len(hyps) < args.beam_size:
            hyps.append(copy.copy(hyps[-1]))

        if steps == args.max_response_steps and not results:
            final_results = sorted_hyps[:args.beam_size]
            # print(type(results))

        hyps = new_hyps

    final_results = []
    for group in results:
        final_results = final_results + group

        # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(final_results)

    # Return the hypothesis with highest average log prob
    return hyps_sorted


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
