import os, copy
import gensim
import numpy as np
import re
import aclImdb

SentenceLoader = aclImdb.SentenceLoader

# **DO NOT CHANGE THE CLASS NAME**
class DataLoader(object):
    def __init__(self, dataset_dir, wordvec_file,
                mode='train', partial_dataset=False, shuffle=False,
                sentence_len=10, wordvec_dim=100, cached=False):
        self.args_to_nn = dict()
        # load sentences
        self.sentences = SentenceLoader(dataset_dir,
                                        with_label=True, full_feature=True,
                                        partial_dataset=partial_dataset,
                                        shuffle=shuffle, mode=mode,
                                        cached=cached)

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
Parser for aclImdb Dataset
Directory structure:
<root>
    - train
        - pos
        - neg
    - test
        - pos
        - neg
        - unsup
'''
