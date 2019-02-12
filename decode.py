import json
import os
import time
import tensorflow as tf
from tqdm import tqdm
import numpy as np

import beam_search
import vocabulary
from metrics.compute_metrics import compute_bleu, compute_rouge, compute_cider, compute_meteor, compute_metrics


class BeamSearchDecoder(object):

    def __init__(self, args, model, batcher, vocab):
        self._args = args
        self._model = model
        self._batcher = batcher
        self._vocab = vocab
        self._saver = tf.train.Saver()  # we use this to load checkpoints for decoding

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=session_config)

        # Load an initial checkpoint to use for decoding
        self._saver.restore(self._sess, args.decode_model)
        self._sess.run(tf.assign(self._model.is_train, tf.constant(False, dtype=tf.bool)))

        self._decode_dir = args.decode_dir
        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir):
            os.makedirs(self._decode_dir)

    def decode(self):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        self._sess.run(tf.assign(self._model.is_train, tf.constant(False, tf.bool)))

        gts, res, weights = {}, {}, {}
        examples = []
        # gts is references dict, res is generate results, weights is references scores list.
        for id, batch in tqdm(enumerate(self._batcher), desc='test'):  # 1 example repeated across batch

            original_query = batch.original_query
            original_description = batch.original_description  # string
            original_responses = batch.original_responses  # string

            # Run beam search to get best Hypothesis
            hyps= beam_search.run_beam_search(self._args, self._sess, self._model, self._vocab, batch)

            # Extract the output ids from the hypothesis and convert back to words
            result = []
            count = 0
            for hyp in hyps:
                output_ids = [int(t) for t in hyp.tokens[1:]]
                decoded_words = vocabulary.outputids2words(output_ids, self._vocab,
                                                           (batch.art_oovs[0] if self._args.pointer_gen else None))

                # Remove the [STOP] token from decoded_words, if necessary
                try:
                    fst_stop_idx = decoded_words.index(vocabulary.STOP_DECODING)  # index of the (first) [STOP] symbol
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    decoded_words = decoded_words
                decoded_output = ' '.join(decoded_words)  # single string



                result.append(decoded_output)

            

            try:
                selected_response = result[0]
                selected_response = vocabulary.response2keywords(selected_response,self._vocab)
                selected_response = ' '.join(selected_response)
            except:
                selected_response = ""

            #gts[id] = original_responses
            #res[id] = [selected_response]
            #weights[id]= original_scores


            # write results to file.
            example = {
                'query': original_query,
                'decription': original_description,
                'responses': original_responses,
                'generate': result,
                'select_cmt': selected_response,
            }
            examples.append(example)

            if id >= 200:
                break

        #self.evaluate(gts, res, weights)
        result_file = os.path.join(self._decode_dir, 'results.json')
        with open(result_file, 'w', encoding='utf8',)as p:
            json.dump(examples, p, indent=2, ensure_ascii=False)

    def evaluate(self, gts, res, weights):
        print(f'{time.asctime()} - Begin to evaluate...')

        bleu_scores = compute_bleu(gts, res)
        rouge_score = compute_rouge(gts, res)
        cider = compute_cider(gts, res)
        metric_dict = {
            'Bleu scores': json.dumps(bleu_scores),
            'Rouge score': rouge_score,
            'CIDEr score': cider,}
        print(json.dumps(metric_dict, indent=2))


if __name__ =='__main__':
    pass

