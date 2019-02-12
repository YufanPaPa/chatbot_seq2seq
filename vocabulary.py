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

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import glob
import random
import struct
import csv
import json

import time
from tensorflow.core.example import example_pb2

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.

class User_Vocab(object):
    def __init__(self, args, name):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [PAD], [UNK], [START] [STOP] ,"low", "medium", "high" get the ids 0,1,2,3
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        for idx in range(50000):
            num_indice = "user_"+str(idx)
            self._word_to_id[num_indice] = self._count
            self._id_to_word[self._count] = num_indice
            self._count +=1
        # print "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1])
        print(
            f'{time.asctime()} - Finished constructing vocabulary of {self._count} total words. Last word added: {self._id_to_word[self._count-1]}')

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]





class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, args, name):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        if name == "encoder_vocab":
            vocab_file = args.vocab_encoder
        if name == "decoder_vocab":
            vocab_file = args.vocab_decoder
            self.split_idx = args.split_idx + 137
        max_size = args.vocab_size
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [PAD], [UNK], [START] [STOP] ,"low", "medium", "high" get the ids 0,1,2,3
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        for idx in range(50):
            num_indice = "<#response_word_num_"+ str(idx) + "#>"
            self._word_to_id[num_indice] = self._count
            self._id_to_word[self._count] = num_indice
            self._count +=1


        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for i, line in enumerate(vocab_f):
                pieces = line.strip().split()
                if len(pieces)!=1:
                    print("error!!,there is a word not good in the vocab!!")
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        f'<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but {w} is')
                if w in self._word_to_id:
                    #continue
                    raise Exception(f'Duplicated word in vocabulary file: {w}')
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    # print "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count)
                    print(
                        f'{time.asctime()} - max_size of vocab was specified as {max_size}; we now have {self._count} words. Stopping reading.')
                    break

        # print "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1])
        print(
            f'{time.asctime()} - Finished constructing vocabulary of {self._count} total words. Last word added: {self._id_to_word[self._count-1]}')







    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def ids2words(self, ids):
        words = []
        for w_id in ids:
            if w_id not in self._id_to_word:
                raise ValueError('Id not found in vocab: %d' % w_id)
            words.append(self._id_to_word[w_id])
        return words

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def write_metadata(self, fpath):
        """Writes metadata file for Tensorboard word embedding visualizer as described here:
          https://www.tensorflow.org/get_started/embedding_viz

        Args:
          fpath: place to write the metadata file
        """
        print(f"Writing word embedding metadata file to {fpath}...")
        with open(fpath, "w", encoding='utf8') as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def news2ids(news_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.

    Args:
      news_words: list of words (strings)
      vocab: Vocabulary object

    Returns:
      ids:
        A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
      oovs:
        A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in news_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_idx = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_idx)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def comment2ids(comment_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in comment_words:
        i = vocab.word2id(w)
        if i==unk_id:
            if w in article_oovs:
                vocab_idx = vocab.size()+article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids



def outputids2words(id_list, vocab, article_oovs):
    """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

    Args:
      id_list: list of ids (integers)
      vocab: Vocabulary object
      article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

    Returns:
      words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words

def response2keywords(selected_response,vocab):
    response = []
    words = selected_response.split()
    for word in words:
        try:
            #print(word)
            idx = vocab.word2id(word)
            if int(idx) > 20857:
                level = vocab._id_to_level[idx]
                word = word+"(Special_Word_"+str(level)+")"
                #print(word)
            response.append(word)
        except:
            continue
    return response

def abstract2sents(abstract):
    """Splits abstract text from datafile into list of sentences.

    Args:
      abstract: string containing <s> and </s> tags for starts and ends of sentences

    Returns:
      sents: List of sentence strings (no tags)"""
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    """Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [f"__{w}__" if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str

def show_title_oovs(title, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = title.split(' ')
    words = [f"__{w}__" if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str

def show_comment_oovs(comment, vocab, article_oovs):

    """Returns the abstract string, highlighting the article OOVs with __underscores__.

    If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.

    Args:
      comment: string
      vocab: Vocabulary object
      article_oovs: list of words (strings), or None (in baseline mode)
    """
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = comment.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append(f"__{w}__")
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append(f"__{w}__")
                else:
                    new_words.append(f"!!__{w}__!!")
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str


