import argparse
import json
import time
import os
import sys
from datetime import datetime
from tqdm import tqdm

from dataset import Dataset, TestBatcher
from vocabulary import Vocab,User_Vocab
# from tensorflow.python import debug as tf_debug
from utils import get_record_parser, get_batch_dataset
import tensorflow as tf
import numpy as np
from model import CommentModel
from decode import BeamSearchDecoder
from tensorflow.contrib.framework.python.framework import checkpoint_utils


def parse_args():
    parser = argparse.ArgumentParser('comment generation')

    parser.add_argument('--mode', type=str, required=True, help='mode of experiment. train/eval/decode/prepare')

    path_settings = parser.add_argument_group('path setting')
    path_settings.add_argument('-d', '--data', type=str, default='../chatbot/data', help='the data file.')
    path_settings.add_argument('-o', '--output', type=str, default='outpout', help='the output dir.')
    path_settings.add_argument('--vocab_encoder', type=str, default='../chatbot/data/vocab/seq2seq_encoder_vocab',
                               help='the encoder vocab file.')
    path_settings.add_argument('--vocab_decoder', type=str, default='../chatbot/data/vocab/seq2seq_decoder_vocab_v2_30597',
                               help='the decoder vocab file.')
    path_settings.add_argument('--special_word_level', type=str, default='../chatbot/data/vocab/special_level.json',
                               help='the special_level file.')
    path_settings.add_argument('--select_user2indice', type=str,default='../chatbot/data/chat_data/select_user.json',
                               help='the description2keyword file.')

    path_settings.add_argument('--records-dir', type=str, default='../chatbot/data/records_big_data_v4', help='the records files.')

    philly_settings = parser.add_argument_group('philly_setting')
    philly_settings.add_argument('--log_dir', type=str, help='the tensorboard log dir.')
    philly_settings.add_argument('--model_dir', type=str, help='the model dir.')
    philly_settings.add_argument('--data_dir', type=str, help='the data dir.')

    model_settings = parser.add_argument_group('model setting')
    model_settings.add_argument('--split_idx', type=int, default=30597)
    model_settings.add_argument('-vsz', '--vocab-size', type=int, default=0)
    model_settings.add_argument('-vsze', '--vocab_size_encoder', type=int, default=100200)
    model_settings.add_argument('-vszd', '--vocab_size_decoder', type=int, default=50200)
    model_settings.add_argument('--hidden-dim', type=int, default=258, help='the hidden dim.')  # 256->128
    model_settings.add_argument('--token-emb-dim', type=int, default=258, help='the embedding dim.')
    model_settings.add_argument('--pos-emb-dim', type=int, default=128, help='the position embedding dim.')
    model_settings.add_argument('--pointer-gen', action='store_true', help='whether use copy mechanism.')
    model_settings.add_argument('--max-query-len', type=int, default=50, help='the max query length.')
    model_settings.add_argument('--max-response-steps', type=int, default=50, help='the max response length.')
    model_settings.add_argument('--trunc-norm-init-std', type=float, default=1e-4,
                                help='std of trunc norm init, used for initializing everything else.')
    model_settings.add_argument('--rand-unif-init-mag', type=float, default=0.02,
                                help='magnitude for lstm cells random uniform initialization.')
    model_settings.add_argument('--dropout-keep-prob', type=float, default=0.8, help='the dropout keep prob.')
    model_settings.add_argument('-b', '--batch-size', type=int, default=20, help='batch size.')
    model_settings.add_argument('--lr', '--learning-rate', type=float, default=0.15, help='learning rate.')
    model_settings.add_argument('-opt', '--optimizer', type=str, default='adagrad', help='optimizer.')
    model_settings.add_argument('--adagrad_init_acc', type=float, default=0.1,
                                help='initial accumulator value for Adagrad')
    model_settings.add_argument('--max-grad-norm', type=float, default=5.0, help='for gradient clip.')  # 2.0->5.0

    train_settings = parser.add_argument_group("train setting")
    train_settings.add_argument('--restore', action='store_true', help='whether restore the latest model.')
    train_settings.add_argument('--restore-model', type=str, default='', help='the restore ckpt.')
    train_settings.add_argument('--period', type=int, default=50, help='period to save the batch loss.')
    train_settings.add_argument('--checkpoint', type=int, default=50000, help='checkpoint for evaluation.')
    train_settings.add_argument('--patience', type=int, default=1, help='the max patience before halve lr.')
    train_settings.add_argument('--no-eval', action='store_true', help='whether evaluate during training.')
    train_settings.add_argument('--gpu', type=str, default='0', help='gpus')

    eval_settings = parser.add_argument_group('eval setting')
    eval_settings.add_argument('--eval-model', type=str, help='the model to evaluation.')

    decode_settings = parser.add_argument_group('decode setting')
    decode_settings.add_argument('--decode-spans', action='store_true', help='decode spans or use original spans.')
    decode_settings.add_argument('--topk', type=int, default=5,
                                 help='When decoding, How many results to give from the model.')
    decode_settings.add_argument('--beam-size', '--beam_size', type=int, default=5, help='beam size for decoding.')
    decode_settings.add_argument('--min-dec-steps', type=int, default=1,
                                 help='Minimum sequence length of generated summary. Applies only for beam search decoding mode.')
    decode_settings.add_argument('--max-query-len-test', type=int, default=50, help='max query len during decode.')
    decode_settings.add_argument('--max-span-seq-len-test', type=int, default=500,
                                 help='max span seq len during decode.')
    decode_settings.add_argument('--decode-model', type=str, required='--decode' in sys.argv,
                                 help='the model for decode.')
    decode_settings.add_argument('--decode-dir', type=str, required='--decode' in sys.argv,
                                 help='the dir store results.')
    decode_settings.add_argument('--cpu', action='store_true', help='use cpu to decode.')
    decode_settings.add_argument('--strict', action='store_true', help='whether use strict strategy.')
    decode_settings.add_argument('--all_vocab_size', type=int, default=100000,
                                 required='--decode' in sys.argv,
                                 help='the user-keyword for seq2seq.')

    return parser.parse_args()


def prepare(args):
    if not os.path.exists(args.records_dir):
        os.makedirs(args.records_dir)

    train_file = os.path.join(args.data, 'chat_data/tmp.data')
    dev_file = os.path.join(args.data, 'chat_data/tmp.data')
    vocab_encoder = Vocab(args, name="encoder_vocab")
    vocab_decoder = Vocab(args, name="decoder_vocab")
    vocab_user = User_Vocab(args, name="user_vocab")
    dataset = Dataset(args, vocab_encoder, vocab_decoder, vocab_user, train_file, dev_file)
    dataset.save_datasets(['train', 'dev'])
    # dataset.save_datasets(['train'])

    # meta_train_file = os.path.join(args.data, 'news_train_span_50.data')
    # dataset = Dataset(args,vocab, meta_train_file, dev_file)
    # dataset.save_datasets(['train'])


def debug(args):
    from utils import get_record_parser, get_batch_dataset
    import tensorflow as tf
    # parser = get_record_parser(args)
    # dataset = get_batch_dataset('data/records/dev.tfrecords', parser, args)
    # iterator = dataset.make_one_shot_iterator()
    # sess = tf.Session()
    # while True:
    #     print(sess.run(iterator.get_next()))
    # break
    # vocab = Vocab(args.vocab, args.vocab_size)
    # test_file = os.path.join(args.data, 'news_test.data')
    # batcher = TestBatcher(args, vocab, test_file).batcher()
    # for b in batcher:
    #     pass

    parser = get_record_parser(args)
    dataset = get_batch_dataset('data/records/dev.tfrecords', parser, args)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    train_iterator = dataset.make_one_shot_iterator()

    model = CommentModel(args, iterator)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    sess = tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())

    train_handle = sess.run(train_iterator.string_handle())

    sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
    get_results = {
        'description': model.description,
        'description_sen': model.description_sen,
        'description_off': model.description_off,
        'description_len': model.description_len,
        'description_mask': model.description_mask,
        'query': model.query,
        'query_len': model.query_len,
        'query_mask': model.query_mask,
        'response': model.response,
        'response_len': model.response_len,
        'response_mask': model.response_mask,
        'target': model.target,
        'target_len': model.target_len,
        'target_mask': model.target_mask,
        # 'x': model.x,
        # 'y': model.y,
        # 'spans': model.span_seq,
        # 'span_num': model.span_num,
        # 'span_mask': model.span_mask
    }

    while True:
        results = sess.run(get_results, feed_dict={handle: train_handle})
        results = {k: v.tolist() for k, v in results.items()}
        from pprint import pprint
        if results['loss'] > 100000:
            pprint(results['loss'], width=1000)
            pprint(results['target'], width=1000)
            pprint(results['target_mask'], width=1000)


def train(args):
    output_dir = args.output
    log_dir = args.log_dir if args.log_dir else os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'model')
    records_dir = args.records_dir if not args.data_dir else os.path.join(args.data_dir, args.records_dir)
    result_dir = os.path.join(output_dir, 'result')
    for dir in [output_dir, log_dir, model_dir, result_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # save the args info to ouptut dir.
    with open(os.path.join(output_dir, 'args.json'), 'w')as p:
        json.dump(vars(args), p, indent=2)

    # load meta info
    with open(os.path.join(records_dir, 'train_meta.json'), 'r', encoding='utf8')as p:
        train_total = json.load(p)['size']
        batch_num_per_epoch = int(np.ceil(train_total / args.batch_size))
        print(f'{time.asctime()} - batch num per epoch: {batch_num_per_epoch}')

    with open(os.path.join(records_dir, 'dev_meta.json'), 'r', encoding='utf8')as p:
        dev_total = json.load(p)['size']

    train_records_file = os.path.join(records_dir, 'train.tfrecords')
    dev_records_file = os.path.join(records_dir, 'dev.tfrecords')

    with tf.Graph().as_default()as graph, tf.device('/gpu:0'):

        parser = get_record_parser(args)
        train_dataset = get_batch_dataset(train_records_file, parser, args)
        dev_dataset = get_batch_dataset(dev_records_file, parser, args)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = CommentModel(args, iterator)

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        sess = tf.Session(config=session_config)

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        writer = tf.summary.FileWriter(log_dir)
        best_ppl = tf.Variable(300, trainable=False, name='best_ppl', dtype=tf.float32)


        saver = tf.train.Saver(max_to_keep=10000)
        if args.restore:
            model_file = args.restore_model or tf.train.latest_checkpoint(model_dir)
            print(f'{time.asctime()} - Restore model from {model_file}..')
            var_list = [_[0] for _ in checkpoint_utils.list_variables(model_file)]
            saved_vars = [_ for _ in tf.global_variables() if _.name.split(':')[0] in var_list]
            res_saver = tf.train.Saver(saved_vars)
            res_saver.restore(sess, model_file)

            left_vars = [_ for _ in tf.global_variables() if _.name.split(':')[0] not in var_list]
            sess.run(tf.initialize_variables(left_vars))
            print(f'{time.asctime()} - Restore {len(var_list)} vars and initialize {len(left_vars)} vars.')
            print(left_vars)
        else:
            print(f'{time.asctime()} - Initialize model..')
            sess.run(tf.global_variables_initializer())
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)

        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())

        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool))) #tmp

        patience = 0

        lr = sess.run(model.lr)
        b_ppl = sess.run(best_ppl)
        print(f'{time.asctime()} - lr: {lr:.3f}  best_ppl:{b_ppl:.3f}')

        t0 = datetime.now()

        while True:
            global_step = sess.run(model.global_step) + 1
            epoch = int(np.ceil(global_step / batch_num_per_epoch))


            loss, loss_gen, ppl, train_op, merge_sum, target, check_1 = sess.run(
                [model.loss, model.loss_gen, model.ppl, model._train_op,
                 model._summaries, model.target,
                 model.check_dec_outputs],
                feed_dict={handle: train_handle})


            ela_time = str(datetime.now() - t0).split('.')[0]

            print((f'{time.asctime()} - step/epoch:{global_step}/{epoch:<3d}   '
                   f'gen_loss:{loss_gen:<3.3f}  '
                   f'ppl:{ppl:<4.3f}  '
                   f'elapsed:{ela_time}\r'), end='')



            if global_step % args.period == 0:
                writer.add_summary(merge_sum, global_step)
                writer.flush()

            if global_step % args.checkpoint == 0:
                model_file = os.path.join(model_dir, 'model')
                saver.save(sess, model_file, global_step=global_step)

            # if  global_step % batch_num_per_epoch== 0:
            if global_step % args.checkpoint == 0 and not args.no_eval:
                sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
                metrics, summ = evaluate_batch(model, dev_total // args.batch_size, sess, handle, dev_handle, iterator)
                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))

                for s in summ:
                    writer.add_summary(s, global_step)

                dev_ppl = metrics['ppl']
                dev_gen_loss = metrics['gen_loss']

                tqdm.write(
                    f'{time.asctime()} - Evaluate after steps:{global_step}, '
                    f' gen_loss:{dev_gen_loss:.4f},  ppl:{dev_ppl:.3f}')

                if dev_ppl < b_ppl:
                    sess.run(tf.assign(best_ppl, dev_ppl))
                    saver.save(sess, save_path=os.path.join(model_dir, 'best'))
                    tqdm.write(f'{time.asctime()} - the ppl is lower than current best ppl so saved the model.')
                    patience = 0
                else:
                    patience += 1

                if patience >= args.patience:
                    lr = lr / 2
                    sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                    patience = 0
                    tqdm.write(f'{time.asctime()} - The lr is decayed form {lr*2} to {lr}.')


def evaluate_batch(model, num_batches, sess, handle, str_handle, iterator, eval_all=True, eval_num=100):
    gen_losses = []
    if not eval_all:
        select_num = eval_num
        indices = np.arange(0, num_batches)
        np.random.shuffle(indices)
        select_indices = indices[:select_num]
    else:
        select_indices = np.arange(0, num_batches)
    for i in tqdm(range(0, num_batches), desc='dev'):
        if i not in select_indices:
            iterator.get_next()
            continue
        gen_loss = sess.run(model.loss_gen, feed_dict={handle: str_handle})
        gen_losses.append(gen_loss)
    gen_loss = np.mean(gen_losses)
    metrics = {}
    metrics['gen_loss'] = gen_loss
    metrics['ppl'] = np.exp(gen_loss)
    gen_loss_summ = tf.Summary(value=[tf.Summary.Value(
        tag='dev/gen_loss', simple_value=metrics['gen_loss']
    )])
    ppl_summ = tf.Summary(value=[tf.Summary.Value(
        tag='dev/ppl', simple_value=metrics['ppl']
    )])
    return metrics, [gen_loss_summ, ppl_summ]


def evaluate(args):
    with open(os.path.join(args.records_dir, 'dev_meta.json'), 'r', encoding='utf8')as p:
        dev_total = json.load(p)['size']

    dev_records_file = os.path.join(args.records_dir, 'dev.tfrecords')
    parser = get_record_parser(args)
    dev_dataset = get_batch_dataset(dev_records_file, parser, args)
    dev_iterator = dev_dataset.make_one_shot_iterator()

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    model = CommentModel(args, dev_iterator)
    saver = tf.train.Saver(max_to_keep=10000)
    saver.restore(sess, args.eval_model)

    total_loss = 0
    batches_num = int(np.ceil(dev_total / args.batch_size))
    for i in tqdm(range(batches_num), desc='eval'):
        loss = sess.run(model.loss)
        total_loss += loss
    dev_loss = total_loss / batches_num
    dev_ppl = np.exp(dev_loss)
    print(f'{time.asctime()} - Evaluation result -> dev_loss:{dev_loss:.3f}  dev_ppl:{dev_ppl:.3f}')


def main(args):
    if args.mode == 'prepare':  # python3 run.py  --mode prepare --pointer-gen
        prepare(args)
    elif args.mode == 'train':  # python3 run.py  --mode train -b 100 -o output --gpu 0  --restore
        train(args)
    elif args.mode == 'eval':
        # python3 run.py --mode eval --eval-model
        evaluate(args)
    elif args.mode == 'decode':  #
        # python3 run.py --mode decode --beam-size 10 --decode-model output_big_data/model/model-250000 --decode-dir output_big_data/result --gpu 1
        args.batch_size = args.beam_size
        vocab_encoder = Vocab(args, "encoder_vocab")
        vocab_decoder = Vocab(args, "decoder_vocab")
        vocab_user = User_Vocab(args, name="user_vocab")
        test_file = "./test.data"
        #test_file = os.path.join(args.data, 'chat_data/tmp.data')
        # test_file = os.path.join(args.data, 'news_train_span_50.data')
        batcher = TestBatcher(args, vocab_encoder, vocab_decoder, vocab_user, test_file).batcher()
        if args.cpu:
            with tf.device('/cpu:0'):
                model = CommentModel(args, vocab_decoder)
        else:
            model = CommentModel(args, vocab_decoder)

        decoder = BeamSearchDecoder(args, model, batcher, vocab_decoder)
        decoder.decode()
    elif args.mode == 'debug':
        debug(args)
    else:
        raise RuntimeError(f'mode {args.mode} is invalid.')


if __name__ == '__main__':
    args = parse_args()
    # which model to use.
    # if args.model == 'model':
    #     from model import CommentModel
    # elif args.model == 'model_g2':
    #     from model_gate import CommentModel
    # elif args.model == 'model_embedding':
    #     from model_embedding import CommentModel
    # elif args.model == 'model_select':
    #     from model_select_gate import CommentModel
    # else:
    #     raise ValueError(f'the arg model {args.model} is invalid.')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(json.dumps(vars(args), indent=2))
    main(args)
