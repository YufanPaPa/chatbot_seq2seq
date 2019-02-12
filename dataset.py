import tensorflow as tf
import json
import os
import sys
import numpy as np
import time
from tqdm import tqdm
from utils import sentence_segment, response_segment
import vocabulary
from vocabulary import news2ids, comment2ids
from natto import MeCab

INF = 1e30


class Example(object):
    def __init__(self, args, vocab_encoder, vocab_decoder, vocab_user):
        self.pointer_gen = args.pointer_gen
        # self.max_article_len = args.max_article_len if args.mode != 'decode' else args.max_article_len_test
        # self.max_title_len = args.max_title_len if args.mode != 'decode' else args.max_title_len_test
        self.max_query_len = args.max_query_len if args.mode != 'decode' else args.max_query_len_test
        # self.max_dec_steps = args.max_dec_steps
        self.max_response_steps = args.max_response_steps
        # self.max_span_num = args.max_span_num
        # self.max_span_len = args.max_span_len
        # self.max_span_seq_len = args.max_span_seq_len if args.mode != 'decode' else args.max_span_seq_len_test
        self.vocab_encoder = vocab_encoder
        self.vocab_decoder = vocab_decoder
        self.vocab_user = vocab_user

        self.pad_id = vocab_encoder.word2id(vocabulary.PAD_TOKEN)
        self.start_id = vocab_encoder.word2id(vocabulary.START_DECODING)
        self.stop_id = vocab_encoder.word2id(vocabulary.STOP_DECODING)
        self.unknown_id = vocab_encoder.word2id(vocabulary.UNKNOWN_TOKEN)

        # self.general_mask = [0 for i in range(args.split_idx+57)]+[-INF for i in range(vocab_decoder.size()-args.split_idx-57)]
        # self.special_mask = [-INF for i in range(args.split_idx+57)]+[0 for i in range(vocab_decoder.size()-args.split_idx-57)]

        self.s_pad_id = 0  # the pad id of sentence offset.
        self.w_pad_id = 0  # the pad id of word offset.
        # self.span_pad_id = 0  # the pad id of x y for append spans.

    def add_description(self, user_string):
        '''
        Add a article to sample.
        Args:
            article: string with space between words.

        Returns:
            None
        '''
        self.user_indice = [self.vocab_user.word2id(user_string)]

        self.original_user_string = user_string
    
    def add_description_decode(self, description):

        description_sentences, description_words, description_ids = [], [], []
        description_split = []
        with MeCab(r'-F%m,%f[0],%h,%f[8]\n -UUNK\n') as nm:
            parse_results = nm.parse(description).strip().split("\n")
            for item in parse_results:
                item = item.split(",")
                if len(item) == 4 and item[1] == "名詞" and len(item[0]) >= 2:
                    description_split.append(item[0])

        for word in description_split:
            if 1 + len(description_words) > self.max_description_len:
                break
            description_sentences.append(word)
            description_words.append(word)
            description_ids.append((self.vocab_decoder.word2id(word), 0, 0))

        self.description_sentences = description_sentences
        self.description_words = description_words
        self.description_ids = description_ids
        #self.description_sentences = []
        #self.description_words = []
        #self.description_ids = []
        self.sentence_num = len(description_sentences)
        self.words_num = len(description_words)

        if self.pointer_gen:
            description_ids_extend_vocab, self.description_oovs = news2ids(description_words, self.vocab_encoder)
            self.description_ids_extend_vocab = [(e_id, s_id, w_id) for e_id, (_, s_id, w_id) in
                                                 zip(description_ids_extend_vocab, description_ids)]
            self.extend_oovs_len = len(self.description_oovs)
            self.extend_oovs = ' '.join(self.description_oovs)

        self.original_description = " ".join(description_words)    

    def add_decode_mask(self, special_word_dict):
        try:
            style_dict = special_word_dict[self.original_description]
        except:
            style_dict = []
            print("we haven't found this description" + str(self.original_description))
        self.original_style_dict = style_dict
        self.style_dict_ids = [self.vocab_decoder.word2id(_) for _ in style_dict]

    def add_query(self, query):
        query_words = query.split()
        response_num_string = "<#response_word_num_" + str(self.response_nums) + "#>"
        response_num_indice = self.vocab_encoder.word2id(response_num_string)

        self.query_ids = [self.vocab_encoder.word2id(_) for _ in query_words]
        self.query_ids.insert(0, response_num_indice)

        self.original_query = query

    def add_query_decode(self, query):
        query_words = query.split()
        response_num_string = "<#response_word_num_"+ str(15) + "#>"
        response_num_indice = self.vocab_encoder.word2id(response_num_string)

        self.query_ids = [self.vocab_encoder.word2id(_) for _ in query_words]
        self.query_ids.insert(0, response_num_indice)
        self.original_query = query

    def add_response(self, response):

        response_words = response.split()
        self.response_nums = len(response_words)
        self.response_ids = [self.vocab_decoder.word2id(_) for _ in response_words]

        self.dec_inp, self.target = self._get_dec_inp_targ_seqs(self.response_ids)

        self.original_response = response

    def add_responses(self, responses: list, scores: list):  # the source format comments list.  used for decode.
        responses_words = [cmt.split() for cmt in responses]
        self.response_ids = [[self.vocab_decoder.word2id(_) for _ in cmt_words] for cmt_words in responses_words]
        self.scores = scores
        self.original_responses = responses

    def _get_dec_inp_targ_seqs(self, sequence):
        inp = [self.start_id] + sequence
        target = sequence + [self.stop_id]
        assert len(inp) == len(target)
        return inp, target

    def pad_tru_description_input(self):
        while len(self.description_ids) < self.max_description_len:
            self.description_ids.append((self.pad_id, self.s_pad_id, self.w_pad_id))
            if self.pointer_gen:
                self.description_ids_extend_vocab.append((self.pad_id, self.s_pad_id, self.w_pad_id))
                assert len(self.description_ids) == len(self.description_ids_extend_vocab)

    def pad_tru_query_input(self):
        if len(self.query_ids) < self.max_query_len:
            while len(self.query_ids) < self.max_query_len:
                self.query_ids.append(self.pad_id)
        else:
            self.query_ids = self.query_ids[:self.max_query_len]

    def pad_tru_decoder_inp_targ(self):
        if len(self.dec_inp) < self.max_response_steps:
            while len(self.dec_inp) < self.max_response_steps:
                self.dec_inp.append(self.pad_id)
            while len(self.target) < self.max_response_steps:
                self.target.append(self.pad_id)


        else:
            self.dec_inp = self.dec_inp[:self.max_response_steps]
            self.target = self.target[:self.max_response_steps]


    def compute_similarity(self, description_embedding, response_embedding):
        if (self.special_word_num != 0):
            try:
                similarity = np.matmul(description_embedding, response_embedding.T)
                (line,coloum) = np.unravel_index(np.argmax(similarity, axis=None), similarity.shape)
                self.similty = np.max(similarity)/(np.linalg.norm(description_embedding[line])*np.linalg.norm(response_embedding[coloum]))
            except:
                self.similty = 0
        else:
            self.similty = 0


class Dataset(object):
    def __init__(self, args, vocab_encoder, vocab_decoder, vocab_user, train_file, dev_file):
        self.args = args
        self.vocab_encoder = vocab_encoder
        self.vocab_decoder = vocab_decoder
        self.vocab_user = vocab_user
        self.pad_id = vocab_encoder.word2id(vocabulary.PAD_TOKEN)
        self.train_file = train_file
        self.dev_file = dev_file
        self.clean_info = {
            'max_oov_scale': 0.3,
            'min_query_len': 3,
            'min_response_len': 3,
            'max_query_len': 50,
            'max_response_len': 50
            # 'min_span_num': 1,
        }
        assert self.pad_id == 0, 'the pad id is not 0'
        select_user2indice = args.select_user2indice
        with open(select_user2indice, encoding='utf8')as p:
            self.select_user2indice = json.load(p)



    def save_datasets(self, sets=None):
        if sets == None:
            sets = ['dev', 'train']
        for set in sets:
            set_file = self.train_file if set == 'train' else self.dev_file
            record_file = os.path.join(self.args.records_dir, f'{set}.tfrecords')
            meta_file = os.path.join(self.args.records_dir, f'{set}_meta.json')
            if set and set_file and record_file and meta_file:
                self._save_dataset(set, set_file, record_file, meta_file)

    def _save_dataset(self, set_name, set_file, record_file, meta_file):
        writer = tf.python_io.TFRecordWriter(record_file)
        print(f'{time.asctime()} - Begin to save {set_name} set..')

        file_size = sum(1 for _ in open(set_file, 'r'))

        # statistic_info
        set_size, max_sen_num, max_offset = 0, 0, 0
        drop_num = 0
        examples = []


        with open(set_file, encoding='utf8')as p:
            for i, line in tqdm(enumerate(p), total=file_size, desc=set_name):


                data = json.loads(line)
                query, description, responses = data['title'], data['content'], data['comment']
                example = Example(self.args, self.vocab_encoder, self.vocab_decoder, self.vocab_user)
                description = description.strip()
                try:
                    user_indice = self.select_user2indice[description]
                except:
                    continue

                example.add_description(user_indice)
                response = responses[0][0]
                example.add_response(response)
                example.pad_tru_decoder_inp_targ()

                example.add_query(query)
                example.pad_tru_query_input()



                '''
                print(example.original_user_string)
                print(example.user_indice)
                print(example.original_query)
                print(example.query_ids)
                print("response_nums : "+str(example.response_nums))
                print(example.original_response)
                print(example.dec_inp)
                print(example.target)
                num += 1
                if num==5:
                    assert 0==1
                '''



                if not self._select_example(example):
                    drop_num += 1
                    continue
                ex = tf.train.Example(features=tf.train.Features(
                    feature={
                        "user_indice": tf.train.Feature(int64_list=tf.train.Int64List(value=example.user_indice)),
                        "query_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example.query_ids)),
                        "dec_inp": tf.train.Feature(int64_list=tf.train.Int64List(value=example.dec_inp)),
                        "target": tf.train.Feature(int64_list=tf.train.Int64List(value=example.target)),
                    }
                ))
                writer.write(ex.SerializeToString())
                set_size += 1


        writer.flush()
        writer.close()

        with open(meta_file, 'w', encoding='utf8')as p:
            set_meta = self.clean_info
            set_meta['size'] = set_size
            set_meta['drop'] = drop_num
            set_meta['max_sen_num'] = max_sen_num
            set_meta['max_offset'] = max_offset
            # set_meta['span_ave_num'] = span_total // file_size
            json.dump(set_meta, p)
            print(f'{time.asctime()} - {set_meta}')

    def _select_example(self, example):
        if len(example.query_ids) == 0 or len(example.response_ids) == 0:
            return False
        if len(example.query_ids) < self.clean_info['min_query_len']:
            return False
        if len(example.dec_inp) < self.clean_info['min_response_len']:
            return False
        if len(example.query_ids) > self.clean_info['max_query_len']:
            return False
        if len(example.dec_inp) > self.clean_info['max_response_len']:
            return False
        return True


def description_embedding_process(description, pretrain_word_embedding):
    words = description.strip().split()
    sentence = "".join(words)
    description_embedding = []
    with MeCab(r'-F%m,%f[0],%h,%f[8]\n -UUNK\n') as nm:
        parse_results = nm.parse(sentence).strip().split("\n")
        for result in parse_results:
            result = result.split(",")
            if len(result) == 4 and result[1] == "名詞" and len(result[0]) > 1 and pretrain_word_embedding.get(result[0],
                                                                                                             -1) != -1:
                description_embedding.append(pretrain_word_embedding.get(result[0]))
    if len(description_embedding)!=0:
        description_embedding = np.stack(description_embedding, axis=0)
    return description_embedding


def response_embedding_process(response, pretrain_word_embedding, vocab_decoder):
    response_embedding = []
    response_words = response.strip().split()
    special_word = [word for word in response_words if vocab_decoder.word2id(word) > vocab_decoder.split_idx]
    for word in special_word:
        if pretrain_word_embedding.get(word, -1) != -1:
            response_embedding.append(pretrain_word_embedding.get(word))
    if len(special_word)!=0:
        response_embedding = np.stack(response_embedding, axis=0)
    return response_embedding

def takesim(elem):
    return elem.similty

def examples_process(examples):
    log_file = open("message.log","w")
    sys.stdout = log_file
    examples.sort(key=takesim,reverse = True)
    for example in examples:
        print("description: "+str(example.original_description))
        print("\n")
        print("query: "+str(example.original_query))
        print("\n")
        print("response: "+str(example.original_response))
        print("\n")
        print("style dict: "+str(example.special_word))
        print("\n")
        print("simility: "+str(example.similty))
        print("\n"+"\n")





class test_Batch(object):
    '''Class representing the batch of test examples for comment generation.'''

    def __init__(self, args, example, vocab):
        self._args = args
        self._pointer_gen = args.pointer_gen
        self._beam_size = args.beam_size
        self._batch_size = args.batch_size
        self._max_query_len = args.max_query_len_test
        self._pad_id = vocab.word2id(vocabulary.PAD_TOKEN)

        self._example = example

        self._init_encoder_seq()
        self._store_orig_strings()

    def _init_encoder_seq(self):
        self._example.pad_tru_query_input()
        self.query_ids = np.tile(np.array(self._example.query_ids), [self._beam_size, 1])
        self.user_indice = np.tile(np.array(self._example.user_indice), [self._beam_size, 1])

        # if hasattr(self._example, 'span_seq'):
        # self._example.pad_tru_span()
        # self.span_seq = np.tile(np.array(self._example.span_seq), [self._beam_size, 1])
        # self.span_seq_len = np.array([self._example.span_seq_len]*self._beam_size, dtype=np.int32)

    def _store_orig_strings(self):
        self.original_description = self._example.original_user_string
        self.original_query = self._example.original_query
        self.original_responses = self._example.original_response
        #self.original_scores = self._example.scores
        #self.original_style_dict = self._example.original_style_dict
        #self.original_style_idx = self._example.style_dict_ids


class TestBatcher(object):
    def __init__(self, args, vocab_encoder, vocab_decoder, vocab_user, test_file=None):
        self._args = args
        self._pointer_gen = args.pointer_gen
        self._beam_size = args.beam_size
        self._batch_size = args.batch_size
        self._max_query_len = args.max_query_len_test
        self._vocab_encoder = vocab_encoder
        self._vocab_decoder = vocab_decoder
        self._vocab_user = vocab_user
        self._pad_id = vocab_encoder.word2id(vocabulary.PAD_TOKEN)

        test_file = test_file if test_file else os.path.join(args.data, 'news_test.data')
        select_user2indice = args.select_user2indice
        with open(select_user2indice, encoding='utf8')as p:
            self.select_user2indice = json.load(p) 
 
        self.dataset = self._load_dataset(test_file)
      


    def _load_dataset(self, test_file):
        print(f"{time.asctime()} - Begin to load dataset from {test_file}..")

        dataset = []
        count = 0

        # statistic info
        max_sen_num, max_off = 0, 0

        file_size = sum(1 for _ in open(test_file))
        with open(test_file, 'r', encoding='utf8')as p:
            for line in tqdm(p, total=file_size, desc='test'):
                data = json.loads(line)
                query, description, response = data['title'], data['content'], data['comment']
                description = description.strip()
                try:
                    user_indice = self.select_user2indice[description]
                except:
                    continue
                responses = [_[0] for _ in response]
                #query = "今日 は 外食 だぁ <context_end> おれ は 毎日 外食 みたい な もん"
                #query = "<context_end> あなた の 仕事 は 何 です か ？"
                #query = "<context_end> あなた は 何 を し て い ます か ？"
                #query = "<context_end> 好き な 食べ物 は 何 でしょ う か ね ?"
                query = "… 泉 (´• ̥̥ ω • ̥̥ `) <context_end> いい加減 諦め て も いい の で は … ? お話 だけ なら いくら でも 付き合い ます よ ?"

                example = Example(self._args, self._vocab_encoder, self._vocab_decoder, self._vocab_user)


                example.add_query_decode(query)
                example.add_description(user_indice)
                example.add_response(responses[0])
                #example.add_query_decode(query)


                # if 'span' in data:
                # example.add_spans(data['span'], is_test=True)
                '''
                print(example.original_description)
                print(example.description_ids)
                print(example.original_query)
                print(example.query_ids)
                print(example.original_responses)
                print(example.response_ids)
                print(example.original_style_dict)
                print(example.style_dict_ids)
                assert 0==1
                '''

                #max_sen_num = example.sentence_num if example.sentence_num > max_sen_num else max_sen_num
                #max_off = example.max_word_offset if example.max_word_offset > max_off else max_off

                dataset.append(example)
                count += 1
        print(f'{time.asctime()} - Dataset size:{len(dataset)} max_sen_num:{max_sen_num} max_offset:{max_off}')
        print("the size of test data is " + str(len(dataset)))
        return dataset

    def batcher(self):
        for ep in self.dataset:
            yield test_Batch(self._args, ep, self._vocab_encoder)


if __name__ == '__main__':
    pass
