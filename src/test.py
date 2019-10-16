import sys, time, os, logging
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from importlib import import_module
from sklearn import metrics

'''
Testing:
Load trained weights from file, and feed them to the model.
Test the data, and report scores.
'''
def test(args): # DO NOT EDIT THIS LINE
    '''
    load the dataset
    '''
    logging.info('Using dataset: %s', args.dataset)
    logging.info('Loading data parser: %s' % args.parser)
    args.wordvec = 'var/wordvec/%s/%s.wv' % (args.dataset, args.wordvec)
    logging.info('Loading word vectors from: %s', args.wordvec)

    parser = import_module('src.parsers.%s' % args.parser)
    data_loader = parser.DataLoader(
        dataset_dir='datasets/%s' % args.dataset,
        wordvec_file=args.wordvec,
        partial_dataset=False,
        sentence_len=args.sentence_len,
        wordvec_dim=args.wordvec_dim,
        mode=args.mode,
        shuffle=False)

    '''
    Load the network and the saved weights
    '''
    logging.info('Loading CNN model: %s' % args.model)
    model_src = import_module('src.models.%s' % args.model)
    convnet = model_src.Model(sentence_len=data_loader.sentence_len,
        wordvec_dim=args.wordvec_dim, dl_args=data_loader.args_to_nn)

    weights_dir = 'var/train/%s.%s' % (args.model, args.dataset)
    if args.load_from is None:
        args.load_from = os.listdir(weights_dir)[0]
    weights_file = '%s/%s.pt' % (weights_dir, args.load_from)
    logging.info('Loading weights from %s', weights_file)
    if args.cuda:
        state_checkpoint = torch.load(weights_file)
    else:
        state_checkpoint = torch.load(weights_file, map_location=lambda storage, loc: storage)
    convnet.load_state_dict(state_checkpoint)

    '''
    pass the data through the network
    '''
    test_start_time = time.time()
    run_tests(convnet=convnet, data_loader=data_loader,
              batch_size=args.batch_size,
              use_cuda=args.cuda, log_interval=args.log_interval)
    test_end_time = time.time()
    logging.info('Total testing time: %f', test_end_time - test_start_time)

'''
Run the tests on the test samples
'''
def run_tests(convnet, data_loader,
              batch_size, use_cuda, log_interval):
    logging.info('Testing: batch_size=%d', batch_size)
    logging.warn('Using CUDA? %s', 'YES' if use_cuda else 'NO')

    data_iter = iter(data_loader)
    num_batches = 1 + (len(data_loader) - 1) / batch_size
    logging.debug('#batches = %d', num_batches)

    if use_cuda:
        convnet.cuda()

    actual, predicted = np.array([]), np.array([])
    freq = [0,0]
    for batch_id in xrange(num_batches):
        # load current batch
        batch_X, batch_Y = [], []
        try:
            for i in xrange(batch_size):
                feature, label = data_iter.next()
                batch_X.append(feature)
                batch_Y.append(label)
        except StopIteration:
            pass

        batch_Y = np.array(batch_Y)

        batch_X = torch.FloatTensor(batch_X)
        if use_cuda: batch_X = batch_X.cuda()
        batch_X = autograd.Variable(batch_X)
        output = convnet.forward(batch_X)

        _, pred = output.max(1)
        predicted = np.append(predicted, [pred.data.cpu().numpy()])
        actual = np.append(actual, [batch_Y])

        # debug
        freq[0] += batch_Y[batch_Y == 0].shape[0]
        freq[1] += batch_Y[batch_Y == 1].shape[0]

        if (batch_id + 1) % log_interval == 0:
            logging.debug('Batch %d done', batch_id+1)
            logging.debug('> pos = %d, neg = %d', freq[1], freq[0])
            freq[0] = 0
            freq[1] = 0

    ### Compute scores
    logging.info('Prediction done. Scores:')
    logging.debug('Actual pos=%d', actual[actual == 1].shape[0])
    logging.debug('Actual neg=%d', actual[actual == 0].shape[0])
    logging.debug('Predicted pos=%d', predicted[predicted == 1].shape[0])
    logging.debug('Predicted neg=%d', predicted[predicted == 0].shape[0])
    logging.info('> accuracy = %f', metrics.accuracy_score(actual, predicted))
    logging.info('> precision = %f', metrics.precision_score(actual, predicted))
    logging.info('> recall = %f', metrics.recall_score(actual, predicted))
    logging.info('> F1 = %f', metrics.f1_score(actual, predicted))
