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
        self.with_label = with_label
        self.full_feature = full_feature

        # test set is labelled
        if mode == 'validate': mode = 'test'

        # load pos samples
        pos_dir = os.path.join(dataset_dir, mode, 'pos')
        pos_files = os.listdir(pos_dir)

        # load neg samples
        neg_dir = os.path.join(dataset_dir, mode, 'neg')
        neg_files = os.listdir(neg_dir)

        # list of files with labels
        self.dataset_files = []
        for fname in pos_files:
            self.dataset_files.append((os.path.join(pos_dir, fname), 1))
        for fname in neg_files:
            self.dataset_files.append((os.path.join(neg_dir, fname), 0))

        # truncate if partial_dataset
        if partial_dataset:
            self.dataset_files = self.dataset_files[:20]

        # load all files at one go
        self.cache = dict()
        if cached:
            self.cached = False
            for fname, label in self.dataset_files:
                self.cache[fname] = self.read_file(fname, label)
        self.cached = cached

        # config
        self.shuffle = shuffle


    # returns a processed sentence/feature, as per the config.
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.dataset_files)

        for fname, label in self.dataset_files:
            for line in self.read_file(fname, label):
                yield line

    def __len__(self):
        return len(self.dataset_files)

    def read_file(self, fname, label):
        if self.cached:
            return self.cache[fname]

        content = None
        with open(fname) as f:
            content = f.read().strip()
            content = self.process_line(content)
            if self.full_feature:
                content_temp = []
                for c in content: content_temp.extend(c)
                content = [content_temp]
            if self.with_label:
                content = map(lambda line: (line, label), content)
        return content

    # return the lines after splitting into words, and filtering.
    def process_line(self, content):
        content = re.sub(r"[^A-Za-z0-9,.!?\']"," ",content)
        content = re.sub(r"n't"," not ",content)
        content = re.sub(r"'nt"," not ",content)
        content = re.sub(r"'","",content)
        content = re.sub(r","," , ",content)
        content = re.sub(r"!"," ! ",content)
        content = re.sub(r"\?"," ? ",content)
        content = re.sub(r"Mr.","Mr",content)
        content = re.sub(r"Mrs.","Mrs",content)
        content = re.sub(r"Ms.","Ms",content)
        content = content.split('.')
        content = [line.split() for line in content]
        return content

# **DO NOT CHANGE THE CLASS NAME**
class DataLoader(object):
    def __init__(self, dataset_dir, wordvec_file,
                mode='train', partial_dataset=False, shuffle=False,
                sentence_len=10, wordvec_dim=100, cached=False):
        self.args_to_nn = None
        # load sentences
        self.sentences = SentenceLoader(dataset_dir,
                                        with_label=True, full_feature=True,
                                        partial_dataset=partial_dataset,
                                        shuffle=shuffle, mode=mode,
                                        cached=cached)

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
            sentence.reverse()
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
