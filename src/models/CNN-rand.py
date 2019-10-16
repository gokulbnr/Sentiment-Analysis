import logging
from importlib import import_module

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

parent_model = import_module('src.models.CNN-static').Model

# **DO NOT CHANGE THE CLASS NAME**
class Model(parent_model):
    def __init__(self, n_classes=2, *args, **kwargs):
        super(Model, self).__init__(n_classes=n_classes, *args, **kwargs)

        wordvec_dim = kwargs.pop('wordvec_dim')
        dl_args = kwargs.pop('dl_args')

        self.n_classes = n_classes
        self.vocab_size = dl_args['vocab_size']
        self.embedding = nn.Embedding(self.vocab_size, wordvec_dim)
        self.embedding.weight.data.normal_(0, 0.05)

    def forward(self, x):
        x = self.embedding(x.long())
        return parent_model.forward(self, x)

helpstr = '''(Version 1.0)
CNN-rand
randomly initialise the word vectors, and learn them also.
'''
