'''
Loads sentences from the dataset
Used to learn the word vectors
Handles preprocesing/filtering
'''
# **DO NOT CHANGE THE CLASS NAME**
class SentenceLoader(object):
    def __init__(self, dataset_dir,
                with_label=False, full_feature=False,
                partial_dataset=True, mode='train',
                shuffle=False, cache=False):
        '''Args for __iter__
        @with_label: return [feature, label] (as array)
        @full_feature: return full feature
        @mode: {train, test, validate}
        @shuffle: Randomly shuffle order of data (on each iteration)
        @cache: load entire dataset into memory
        @partial_dataset: load only a part of the data set (debugging)
        '''
        pass

    # returns a processed sentence/feature, as per the config.
    def __iter__(self):
        yield

'''
Loads sentences, and trained word vectors,
and converts the data into wordvectors, by concatenation.
Returns the full vector, with its label.
'''
# **DO NOT CHANGE THE CLASS NAME**
class DataLoader(object):
    def __init__(self, dataset_dir, wordvec_file=None,
                mode='train', partial_dataset=False, shuffle=False,
                sentence_len=10, wordvec_dim=100, cache=False):
        '''
        Load the wordvectors from @wordvec_dir
        modes: train, validate, test
        '''
        # self.sentences = SentenceLoader(dataset_dir,
                                        # with_label=True, full_feature=True,
                                        # mode=mode, cache=cache,
                                        # partial_dataset
                                        # shuffle=shuffle)
        pass

    # returns [x, y]: feature and label
    def __iter__(self):
        yield


helpstr = '''(Version 1.0)
Parser for <dataset>
[Give info here.]
'''
