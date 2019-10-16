from torch import nn
import numpy as np

# **DO NOT CHANGE THE CLASS NAME**
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        '''
        Add layers here
        '''

    def forward(self, x):
        '''
        Pass x through the CNN
        '''
        return x

    def num_flat_features(self, x):
        return reduce(lambda a, b: a * b, x.size()[1:])

helpstr = '''(Version 1.0)
Example model CNN
@input: any Tensor
@returns: @input tensor
'''
