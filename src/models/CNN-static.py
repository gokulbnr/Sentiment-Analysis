import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# **DO NOT CHANGE THE CLASS NAME**
class Model(nn.Module):
    def __init__(self, n_classes=2,
        sentence_len=100, wordvec_dim=100,
        filter_sizes=[(3, 100), (4, 100), (5, 100)],
        dl_args=None):

        super(Model, self).__init__()
        self.n_classes = n_classes

        # Convolution layer and max pooling
        self.convs = []
        self.maxpools = []
        fc_size = 0 # input size to fully connected layer
        for filter_width, num_filters in filter_sizes:
            self.convs.append(
                # (input_channels, output_channels, (filter_width, wordvec_dim))
                nn.Conv2d(1, num_filters, (filter_width, wordvec_dim))
            )
            self.maxpools.append(
                nn.MaxPool1d(sentence_len - filter_width + 1)
            )
            fc_size += num_filters

        self.convs = nn.ModuleList(self.convs)
        self.maxpools = nn.ModuleList(self.maxpools)

        # NN
        layer_sizes = [fc_size, 300]
        self.FC1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.dropout1 = nn.Dropout(p = 0.5)
        self.layer = nn.Linear(layer_sizes[1], 2)

        # random-initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, (2. / n) ** .5)



    def forward(self, x):
        '''
        Pass x through the CNN
        '''
        # x: N * W * D
        # N = batch size, W = no of words, D = wordvec dimension
        # NF = num filters, Ff = num per filter, Fs = filter size

        max_x = []
        for i in range(len(self.convs)):
            curr = x
            curr = x.unsqueeze(1)
            curr = self.convs[i](curr)
            curr = F.relu(curr).squeeze(3)
            curr = self.maxpools[i](curr).squeeze(2)
            max_x.append(curr)
        x = torch.cat(max_x, 1)

        x = x.view(x.size(0), -1) # make it a single row
        # x = self.FC1(x)
        x = self.dropout1(x)
        x = self.layer(x)
        # x = F.softmax(x)
        return x

    def num_flat_features(self, x):
        return reduce(lambda a, b: a * b, x.size()[1:])

helpstr = '''(Version 1.0)
CNN-static
Uses pre-trained word vectors
'''
