import os, copy
import gensim
import numpy as np
import re

# **DO NOT CHANGE THE CLASS NAME**
class SentenceLoader(object):
    def __init__(self, dataset_dir, with_label=False, full_feature=False, 
                 partial_dataset=False, mode='train', shuffle=False, cached=False):
        '''Args for __iter__
        @with_label: return [feature, label] (as array)
        @full_feature: return full feature
        '''
        self.with_label = with_label
        self.full_feature = full_feature

        # test set is labelled
        if mode == 'validate': mode = 'test'

        # load samples
        dataset_files = [(os.path.join(dataset_dir,mode,'rt-polarity.neg'), 0),
                         (os.path.join(dataset_dir,mode,'rt-polarity.pos'), 1)]

        self.dataset_samples = []
        for (fname, label) in dataset_files:
            with open(fname) as f:
                lines = f.read().strip().split('\n')
                for line in lines:
                    sample = self.process_line(line)
                    if self.with_label:
                        sample = (sample, label)
                    self.dataset_samples.append(sample)

        # config
        self.shuffle = shuffle

    # returns a processed sentence/feature, as per the config.
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.dataset_samples)

        for sample in self.dataset_samples:
            yield sample

    def __len__(self):
        return len(self.dataset_samples)

    # return the lines after splitting into words, and filtering.
    def process_line(self, content):
        content = re.sub(r"[^A-Za-z0-9,.!?\']"," ",content)
        content = re.sub(r"n't"," not ", content)
        return content.split()

# **DO NOT CHANGE THE CLASS NAME**
class DataLoader(object):
    def __init__(self, dataset_dir, wordvec_file,
                mode='train', partial_dataset=False, shuffle=False,
                sentence_len=10, wordvec_dim=100, cached=False):
        # load sentences
        self.sentences = SentenceLoader(dataset_dir,
                                        with_label=True,
                                        full_feature=True,
                                        partial_dataset=partial_dataset,
                                        shuffle=shuffle)

        # load the word vectors
        model = gensim.models.Word2Vec.load(wordvec_file)
        self.word_vectors = model.wv # no updates
        del model

        # config
        self.sentence_len = sentence_len
        self.wordvec_dim = wordvec_dim
        self.partial_dataset = partial_dataset
        self.dataset_dir = dataset_dir

    # returns [x, y]: feature and label
    def __iter__(self):
        for sentence, label in self.sentences:
            # sentence.reverse()
            wordvec = np.ndarray((0, self.wordvec_dim))
            count = 0 # only add `self.sentence_len` words
            for word in sentence:
                if word in self.word_vectors.vocab:
                    wordvec = np.append(wordvec, [self.word_vectors[word]], axis=0)
                    count += 1
                    if count == self.sentence_len:
                        break

            # pad with zeros, if sentence is too small
            if count < self.sentence_len:
                wordvec = np.append(wordvec, np.zeros((self.sentence_len - count, self.wordvec_dim)), axis=0)
            yield wordvec, label

    def __len__(self):
        return len(self.sentences)

helpstr = '''(Version 1.0)
Parser for rt Dataset
Directory structure:
<root>
    -train
        - rt-polarity.pos
        - rt-polarity.neg
    -test
        - rt-polarity.pos
        - rt-polarity.neg
'''
