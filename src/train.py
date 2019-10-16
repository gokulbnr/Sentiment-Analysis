import sys, time, os, logging
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from importlib import import_module

'''
Training:
- Use the parser class to load the data, and feed it to the model.
- Log training scores and loss (to `var/log/...`)
- Store weights to `var/train/...`
- Checkpoint weights at regular intervals
'''
def train(args): # DO NOT EDIT THIS LINE
    '''
    load the dataset
    '''
    logging.info('Using dataset: %s', args.dataset)
    args.wordvec = 'var/wordvec/%s/%s.wv' % (args.dataset, args.wordvec)
    logging.info('Loading word vectors: %s', args.wordvec)
    logging.info('Loading data parser: %s' % args.parser)
    parser = import_module('src.parsers.%s' % args.parser)
    data_loader = parser.DataLoader(
        dataset_dir='datasets/%s/' % args.dataset,
        wordvec_file=args.wordvec,
        partial_dataset=args.partial_dataset,
        shuffle=True,
        sentence_len=args.sentence_len,
        cached=args.cache)

    '''
    Load the model, and setup cuda, if needed
    '''
    logging.info('Loading CNN model: %s' % args.model)
    model_src = import_module('src.models.%s' % args.model)
    convnet = model_src.Model(sentence_len=args.sentence_len,
        wordvec_dim=args.wordvec_dim, dl_args=data_loader.args_to_nn)

    # continue from checkpoint?
    if args.load_from is not None:
        weights_file = 'var/train/%s.%s/%s.pt' % (args.model, args.dataset, args.load_from)
        logging.info('Loading weights from %s', weights_file)
        state_checkpoint = torch.load(weights_file)
        convnet.load_state_dict(state_checkpoint)

    '''
    Train the model
    '''
    train_start_time = time.time()
    train_model(convnet=convnet,
                data_loader=data_loader,
                epochs=args.epochs, use_cuda=args.cuda,
                batch_size=args.batch_size,
                train_dir='var/train/%s.%s/' % (args.model, args.dataset),
                job_name=args.job_name,
                log_interval=args.log_interval)
    train_end_time = time.time()
    logging.info('Total training time: %f', train_end_time - train_start_time)

'''
Trains the CNN, with multiple epochs
'''
def train_model(convnet, data_loader, epochs,
                batch_size, use_cuda,
                train_dir, job_name,
                log_interval=10):
    logging.info('Learning: epochs=%d, batch_size=%d', epochs, batch_size)
    logging.warn('Using CUDA? %s', 'YES' if use_cuda else 'NO')

    # Loss function and optimizer for the learning
    loss_func = torch.nn.CrossEntropyLoss()
    learning_rate = 1.0
    decay_rate = 0.8
    decay_interval = 10
    optimizer = optim.Adadelta(convnet.parameters(), lr=learning_rate)

    if use_cuda:
        convnet = convnet.cuda()
        loss_func = loss_func.cuda()

    # check directory for saving train weights
    if not os.path.isdir(train_dir): os.mkdir(train_dir)

    num_batches = 1 + (len(data_loader) - 1) / batch_size
    logging.debug('#batches = %d', num_batches)

    total_data_load_time = 0

    for epoch in xrange(epochs):
        epoch_start_time = time.time()
        epoch_data_load_time = 0

        epoch_loss = 0

        data_iter = iter(data_loader)

        for batch_id in xrange(num_batches):
            # load current batch
            batch_load_start = time.time()
            batch_X, batch_Y = [], []
            try:
                for i in xrange(batch_size):
                    feature, label = data_iter.next()
                    batch_X.append(feature)
                    batch_Y.append(label)
            except StopIteration:
                pass
            batch_load_end = time.time()
            epoch_data_load_time += batch_load_end - batch_load_start

            # make the batch feature variable
            batch_X = torch.FloatTensor(batch_X)
            if use_cuda: batch_X = batch_X.cuda()
            batch_X = autograd.Variable(batch_X)

            # forward pass
            optimizer.zero_grad()
            output = convnet.forward(batch_X)
            if use_cuda: output = output.cuda()

            # compute loss, and backward pass
            batch_Y = torch.FloatTensor(batch_Y)
            if use_cuda: batch_Y = batch_Y.cuda()
            batch_Y = autograd.Variable(batch_Y).long()
            loss = loss_func(output, batch_Y)
            if use_cuda: loss.cuda()
            loss.backward()
            optimizer.step()

            # update total loss
            epoch_loss += loss.data[0]

            # logging
            if (batch_id + 1) % log_interval == 0:
                logging.debug('Batch %d: loss = %f', batch_id + 1, epoch_loss / batch_id)

            # cleanup
            del batch_X, batch_Y
            del output, loss

        epoch_end_time = time.time()

        # save weights
        if (epoch + 1) % log_interval == 0:
            logging.info('Saving weights at epoch %d', epoch + 1)
            save_file = os.path.join(train_dir, '%s_backup_%d.pt' % (job_name, (epoch + 1) / log_interval))
            torch.save(convnet.state_dict(), save_file)

        ### log epoch execution statistics
        logging.info('Epoch %d: time = %.3f', epoch, epoch_end_time - epoch_start_time)
        logging.info('> data load time = %.3f', epoch_data_load_time)
        total_data_load_time += epoch_data_load_time
        logging.info('> Loss = %f', epoch_loss / num_batches)

        # decay learning rate
        if (epoch + 1) % decay_interval == 0:
            learning_rate *= decay_rate
            optimizer = optim.Adadelta(convnet.parameters(), lr=learning_rate)

    # save final trained weights
    logging.info('Saving final weights')
    save_file = os.path.join(train_dir, '%s_final.pt' % job_name)
    torch.save(convnet.state_dict(), save_file)

    logging.info('> Total data load time = %.3f', total_data_load_time)
