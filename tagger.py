import dynet as dy
import numpy as np

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def read(fname):
    """
    Read a POS-tagged file where each line is of the form "word1/tag2 word2/tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    with file(fname) as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("/",1)) for x in line]
            yield sent


class Tagger:
    def __init__(self, train):
        # Train is a list of sentences
        # Sentences are a list of (w,p) tuples

        #self.train = list(read(train_file))

        words,tags,chars = [],[],set()
        wc=Counter()
        for sent in train:
            for w,p in sent:
                words.append(w)
                tags.append(p)
                chars.update(w)
                wc[w] += 1
        words.append("_UNK_")
        chars.add("<*>")
        
        self.words = words
        self.tags = tags
        self.chars = chars

        vw = Vocab.from_corpus([words])
        vt = Vocab.from_corpus([tags])
        vc = Vocab.from_corpus([chars])
        self.nwords = vw.size()
        self.ntags = vt.size()
        self.nchars = vc.size()


        # DyNet Starts
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(model)
        
        self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.nwords, 128))
        self.CHARS_LOOKUP = self.model.add_lookup_parametrs((self.nchars, 20))
        p_t1 = self.model.add_lookup_parametrs((self.ntags, 30))
        
        
        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        pH = self.model.add_parameters((32, 50*2))
        pO = self.model.add_parameters((self.ntags, 32))
        
        # word-level LSTMs
        fwdRNN = dy.LSTMBuilder(1, 128, 50, self.model) # layers, in-dim, out-dim, model
        bwdRNN = dy.LSTMBuilder(1, 128, 50, self.model)
        
        # char-level LSTMs
        cFwdRNN = dy.LSTMBuilder(1, 20, 64, self.model)
        cBwdRNN = dy.LSTMBuilder(1, 20, 64, self.model)
        
        
        

    def train(self, epochs=50):
        best_accuracy = 0
        num_tagged = cum_loss = 0
        num_words = num_spines = 0


    def tag_sent(words):
    def test(self, test_file):
        

        


    def read_dev(self, dev_file):
        self.dev = list(read(dev_file))
    def read_test(self, test_file):
        self.test = list(read(test_file))
