#!/usr/bin/python
import os, sys, logging, logging.config, yaml
import argparse
from importlib import import_module

def setup_logging(args):
    config = {}
    with open('logging.yaml') as f:
        config = yaml.safe_load(f)
        log_file = 'var/log/%s/' % args.task
        if args.task == 'preprocess':
            log_file += args.parser
        else:
            log_file += args.model + '.' + args.dataset
        log_file += '.log'
        config['handlers']['file_handler']['filename'] = log_file
    logging.config.dictConfig(config)

def check_directory_structure():
    dirs = ['var', 'var/train', 'var/wordvec', 'var/log', 'var/log/train', 'var/log/test', 'var/log/preprocess', 'datasets']
    for d in dirs:
        if not os.path.isdir(d):
            logging.warn("Directory `%s` not found, creating." % d)
            os.mkdir(d)


def parse_args(args=None):
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Sentence Classification', formatter_class=ADHF)
    subparsers = parser.add_subparsers(title='Tasks', dest='task')

    # common args
    common_parser = argparse.ArgumentParser(add_help=False)
    common_args = common_parser.add_argument_group('Config')
    common_args.add_argument('--model', default=None,
        help='Model to use (src/models/MODEL.py)')
    common_args.add_argument('--dataset', default=None,
        help='Dataset to use (dataset/DATASET/)')
    common_args.add_argument('--parser', default=None,
        help='Data parser to use (stored in src/parsers/PARSER.py)')
    common_args.add_argument('--no-cuda', dest='cuda', action='store_false',
        help='Do not use CUDA even if available')
    common_args.add_argument('--log-interval', type=int, default=10,
        help='Interval for logging')

    common_config = common_parser.add_argument_group('Data Config')
    common_config.add_argument('--sentence-len', type=int, default=20,
        help='Consider only last <number> words')
    common_config.add_argument('--wordvec-dim', type=int, default=100,
        help='Size of word vector')
    common_config.add_argument('--partial-dataset', action='store_true',
        help='Use only a part of the dataset (for debugging)')

    # Preprocess: generate word vectors
    parser_preprocess = subparsers.add_parser('preprocess', help='Generate word vectors', parents=[common_parser], formatter_class=ADHF)
    parser_preprocess.add_argument('-w', '--workers', type=int, default=4,
        help='Number of workers for gensim')
    parser_preprocess.add_argument('--num-iter', type=int, default=5,
        help='Number of passes')
    parser_preprocess.add_argument('--min-count', type=int, default=5,
        help='Minimum occurences of words')
    parser_preprocess.add_argument('--save-vocab', default='words',
        help='Save the vocabulary to file')
    parser_preprocess.add_argument('-o', '--output', default='model',
        help='File to store learned word vectors.')

    # Train: train the CNN
    parser_train = subparsers.add_parser('train', help='Train the CNN', parents=[common_parser], formatter_class=ADHF)
    parser_train.add_argument('job_name',
        help='Name of the job. Used for logging/checkpointing.')
    parser_train.add_argument('--epochs', type=int, default=100,
        help='Number of epochs to run')
    parser_train.add_argument('--batch-size', type=int, default=100,
        help='Mini-batch size for the CNN')
    parser_train.add_argument('--cache', action='store_true',
        help='Load the entire dataset to memory (use with caution)')
    parser_train.add_argument('--load-from', default=None,
        help='Load weights from file (checkpoint)')
    parser_train.add_argument('--wordvec', default=None,
        help='Load word vectors from file')

    # Test: test the CNN
    parser_test = subparsers.add_parser('test', help='Use the CNN', parents=[common_parser], formatter_class=ADHF)
    parser_test.add_argument('--batch-size', type=int, default=100,
        help='Mini-batch size for the CNN')
    parser_test.add_argument('--load-from', default=None,
        help='Load weights from file (checkpoint)')
    parser_test.add_argument('--wordvec', default=None,
        help='Load word vectors from file')
    parser_test.add_argument('--mode-alias', dest='mode', default='test',
        help='An alias for the test mode.')

    # parse
    return parser.parse_args(args)

def main():
    check_directory_structure()
    if len(sys.argv) >= 3 and sys.argv[1] == '--config':
        # load from file
        with open(sys.argv[2]) as f:
            lines = map(lambda line: line.strip().split(), f.readlines())
            args = []
            for line in lines: args.extend(line)
        args = parse_args(args)
    else:
        args = parse_args()
    setup_logging(args)

    if args.cuda:
        import torch
        args.cuda = torch.cuda.is_available()
        logging.info("CUDA Available: %s", 'YES' if args.cuda else 'NO')

    if args.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        logging.info("Using CUDA.")

    try:
        if args.task == 'preprocess':
            import_module('src.preprocess').preprocess(args)
        elif args.task == 'train':
            import_module('src.train').train(args)
        elif args.task == 'test':
            import_module('src.test').test(args)
    except KeyboardInterrupt:
        logging.warn('Force quit!')
        pass
    except:
        raise
    logging.info('Exiting...\n%s', '-' * 80)

main()
