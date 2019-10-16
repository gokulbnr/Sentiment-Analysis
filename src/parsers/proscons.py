'''
Loads sentences from the dataset
Used to learn the word vectors
Handles preprocesing/filtering
'''
from bs4 import BeautifulSoup
import os, copy
import gensim
import numpy as np
import re

# **DO NOT CHANGE THE CLASS NAME**
class SentenceLoader(object):
    def __init__(self, dataset_dir,
                with_label=False, full_feature=False,
                partial_dataset=False, mode='train',
                shuffle=False, cached=False):
        '''Args for __iter__
        @with_label: return [feature, label] (as array)
        @full_feature: return full feature
        @mode: {train, test, validate}
        @shuffle: Randomly shuffle order of data (on each iteration)
        @cached: load entire dataset into memory
        @partial_dataset: load only a part of the data set (debugging)
        '''
        self.dataset = []
        with open(dataset_dir + '/IntegratedPros.txt') as f:
            lines = f.readlines()
            num_lines = len(lines)
            num_lines = (num_lines / 10) * 7 # 70-30 train-validate
            if mode == 'train':
                lines = lines[:num_lines]
            else:
                lines = lines[num_lines:]

            self.dataset += map(lambda line: (self.process_line(line), 1), lines)
        with open(dataset_dir + '/IntegratedCons.txt') as f:
            lines = f.readlines()
            self.dataset += map(lambda line: (self.process_line(line), 0), lines)

        self.shuffle = shuffle
        self.with_label = with_label

    # returns a processed sentence/feature, as per the config.
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.dataset)
        for line in self.dataset:
            if self.with_label:
                yield line
            else:
                yield line[0]

    def process_line(self, line):
        line = BeautifulSoup(line).text
        line = line.replace(',', ' , ')
        line = line.replace('.', ' . ')
        line = line.strip().split()
        line = [word.strip() for word in line]
        return line

    def __len__(self):
        return len(self.dataset)


'''
Loads sentences, and trained word vectors,
and converts the data into wordvectors, by concatenation.
Returns the full vector, with its label.
'''
# **DO NOT CHANGE THE CLASS NAME**
class DataLoader(object):
    def __init__(self, dataset_dir, wordvec_file=None,
                mode='train', partial_dataset=False, shuffle=False,
                sentence_len=10, wordvec_dim=100, cached=False):
        '''
        Load the wordvectors from @wordvec_dir
        modes: train, validate, test
        '''
        self.sentences = SentenceLoader(dataset_dir,
                                        with_label=True, full_feature=True,
                                        mode=mode, cached=cached,
                                        partial_dataset=False,
                                        shuffle=shuffle)

        # load the word vectors
        model = gensim.models.Word2Vec.load(wordvec_file)
        word_vectors = model.wv # no updates
        del model

        self.vocab = dict()
        idx = 0
        for key in word_vectors.vocab:
            self.vocab[key] = idx
            idx += 1
        self.vocab_size = idx
        self.args_to_nn = dict()
        self.args_to_nn['vocab_size'] = self.vocab_size
        self.args_to_nn['vocab_indices'] = self.vocab
        self.args_to_nn['pre_trained'] = word_vectors

        # config
        self.sentence_len = sentence_len
        self.partial_dataset = partial_dataset
        self.dataset_dir = dataset_dir

    # returns [x, y]: feature and label
    def __iter__(self):
        for sentence, label in self.sentences:
            sentence = sentence[::-1]
            wordvec = [0] * self.sentence_len
            index = 0
            for word in sentence:
                if word in self.vocab:
                    wordvec[index] = self.vocab[word]
                    index += 1
                if index == self.sentence_len:
                    break
            yield wordvec, label

    def __len__(self):
        return len(self.sentences)


helpstr = '''(Version 1.0)
Pros-Cons Dataset
<root>
    - IntegratedCons.txt
    - IntegratedPros.txt
'''
