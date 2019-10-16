import logging
from importlib import import_module

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

parent_model = import_module('src.models.CNN-rand').Model

# **DO NOT CHANGE THE CLASS NAME**
class Model(parent_model):
    def __init__(self, n_classes=2, *args, **kwargs):
        super(Model, self).__init__(n_classes=n_classes, *args, **kwargs)

        # initialise with pre-trained weights
        dl_args = kwargs['dl_args']
        pre_trained_wv = dl_args['pre_trained']
        embed_init = [None] * self.vocab_size
        vocab = kwargs['dl_args']['vocab_indices']
        for word in vocab:
            index = vocab[word]
            embed_init[index] = pre_trained_wv[word]
        embed_init = np.array(embed_init)
        embed_init = torch.FloatTensor(embed_init)
        self.embedding.weight = nn.Parameter(embed_init)


helpstr = '''(Version 1.0)
CNN-non-static
initialise the word vectors with pre-trained weights, but learn them too.
'''
