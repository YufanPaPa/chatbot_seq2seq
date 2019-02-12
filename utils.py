import re
import tensorflow as tf
import json

def sentence_segment(paragraph):
    for sent in re.findall(u'[^！？。]+[！？。]?', paragraph, flags=re.U):
        yield sent

def response_segment(response):
    response = "".join(response.strip().split())
    pattern = r'[,.;?!，。、；!…]+'
    return len(re.split(pattern, response))


def get_record_parser(args):
    def parse(example):
        max_query_len = args.max_query_len
        max_dec_steps = args.max_response_steps
        # max_span_num = args.max_span_num
        # max_span_seq_len = args.max_span_seq_len
        pointer_gen = args.pointer_gen  # whether use copy mechanism

        feas = {
            "user_indice": tf.FixedLenFeature([1],tf.int64),
            "query_ids": tf.FixedLenFeature([max_query_len], tf.int64),
            "dec_inp": tf.FixedLenFeature([max_dec_steps], tf.int64),
            "target": tf.FixedLenFeature([max_dec_steps], tf.int64)
        }

        features = tf.parse_single_example(
            example,
            features=feas
        )
        for k, v in features.items():
            if v.dtype == tf.int64:
                features[k] = tf.cast(v, tf.int32)

        user_indice = features['user_indice']
        query_ids = features['query_ids']
        dec_inp = features['dec_inp']
        target = features['target']



        return (user_indice, query_ids, dec_inp, target)

    return parse


def get_batch_dataset(record_file, parser, args):
    num_threads = tf.constant(8, dtype=tf.int32)
    capacity = 15000
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(
        capacity).repeat()
    dataset = dataset.batch(args.batch_size)
    return dataset


if __name__ == '__main__':
    pass
