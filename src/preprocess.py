import sys, os, logging
from importlib import import_module
import gensim

'''
Preprocessing:
@arg dataset Load from `datasets/@dataset`
@arg parser_name Use parser src/parsers/@parser_name
@arg output_dir Save word vectors to `var/wordvec/@output_dir`
'''
def preprocess(args):
    # load the parser and dataset
    logging.info('Using Parser %s', args.parser)
    parser = import_module('src.parsers.%s' % args.parser)

    logging.info('Loading Dataset: %s', args.dataset)
    if args.partial_dataset: logging.debug('* using only partial dataset *')
    sentences = parser.SentenceLoader(dataset_dir='datasets/%s' % args.dataset,
                                      partial_dataset=args.partial_dataset)

    # learn the word vectors
    model = gensim.models.Word2Vec(sentences,
        min_count=args.min_count, size=args.wordvec_dim,
        workers=args.workers, iter=args.num_iter)

    # save the word vectors
    output_dir = 'var/wordvec/' + args.dataset
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    output_file = os.path.join(output_dir, args.output + '.wv')
    model.save(output_file)

    # save vocab
    if args.save_vocab is not None:
        vocab_file = os.path.join(output_dir, args.save_vocab + '.vocab')
        logging.info('Saving vocabulary to %s', vocab_file)
        with open(vocab_file, 'w') as f:
            for key in model.wv.vocab:
                f.write(key + '\n')

    logging.info('Word Vectors generated!')
