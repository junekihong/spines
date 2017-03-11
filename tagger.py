from collections import Counter, defaultdict
from itertools import count
import random
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

        if "=" not in w2i:
            w2i["="]
        if "@" not in w2i:
            w2i["@"]

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

def strip_spines(tags):
    result = []
    for t in tags:
        if "-" in t and (t != "-LRB-" and t != "-RRB-"):
            t = t.split("-")[0]
        result.append(t)
    return result


class Tagger:
    def __init__(self, train, dev, model_file=None):
        # Train is a list of sentences
        # Sentences are a list of (w,p) tuples

        #self.train = list(read(train_file))
        self.train = train
        self.dev = dev
        self.model_file = model_file if model_file is not None else "model"

        words,tags,chars = [],[],set()
        self.wc=Counter()
        for sent in train:
            for w,p in sent:
                words.append(w)
                tags.append(p)
                chars.update(w)
                self.wc[w] += 1
        words.append("_UNK_")
        chars.add("<*>")
        
        self.words = words
        self.tags = tags
        self.chars = chars

        self.vw = Vocab.from_corpus([words])
        self.vt = Vocab.from_corpus([tags])
        self.vc = Vocab.from_corpus([chars])
        self.nwords = self.vw.size()
        self.ntags = self.vt.size()
        self.nchars = self.vc.size()


        # DyNet Starts
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        
        self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.nwords, 128))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.nchars, 20))
        p_t1 = self.model.add_lookup_parameters((self.ntags, 30))
        
        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        self.pH = self.model.add_parameters((32, 50*2))
        self.pO = self.model.add_parameters((self.ntags, 32))
        
        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, 128, 50, self.model) # layers, in-dim, out-dim, model
        self.bwdRNN = dy.LSTMBuilder(1, 128, 50, self.model)
        
        # char-level LSTMs
        self.cFwdRNN = dy.LSTMBuilder(1, 20, 64, self.model)
        self.cBwdRNN = dy.LSTMBuilder(1, 20, 64, self.model)


    def load_model(self):
        self.model.load(self.model_file)
    def save_model(self):
        self.model.save(self.model_file)


    def word_rep(self, w, cf_init, cb_init):
        if self.wc[w] > 5:
            w_index = self.vw.w2i[w]
            return self.WORDS_LOOKUP[w_index]
        else:
            pad_char = self.vc.w2i["<*>"]
            char_ids = [pad_char] + [self.vc.w2i[c] for c in w] + [pad_char]
            char_embs = [self.CHARS_LOOKUP[cid] for cid in char_ids]
            fw_exps = cf_init.transduce(char_embs)
            bw_exps = cb_init.transduce(reversed(char_embs))
            return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])


    def build_tagging_graph(self, words):
        dy.renew_cg()
        # parameters -> expressions
        H = dy.parameter(self.pH)
        O = dy.parameter(self.pO)
        
        # initialize the RNNs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()
        cf_init = self.cFwdRNN.initial_state()
        cb_init = self.cBwdRNN.initial_state()
        
        # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        wembs = [self.word_rep(w, cf_init, cb_init) for w in words]
        wembs = [dy.noise(we,0.1) for we in wembs] # optional
        
        # feed word vectors into biLSTM
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))
        # OR
        #    fw_exps = []
        #    s = f_init
        #    for we in wembs:
        #        s = s.add_input(we)
        #        fw_exps.append(s.output())
        #    bw_exps = []
        #    s = b_init
        #    for we in reversed(wembs):
        #        s = s.add_input(we)
        #        bw_exps.append(s.output())
        
        # biLSTM states
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]
        
        # feed each biLSTM state to an MLP
        exps = []
        for x in bi_exps:
            r_t = O*(dy.tanh(H * x))
            exps.append(r_t)
        return exps

        
    def sent_loss(self, words, tags):
        vecs = self.build_tagging_graph(words)
        errs = []
        for v,t in zip(vecs,tags):
            tid = self.vt.w2i[t]
            err = dy.pickneglogsoftmax(v, tid)
            errs.append(err)
        return dy.esum(errs)

    def tag_sent(self, words):
        vecs = self.build_tagging_graph(words)
        vecs = [dy.softmax(v) for v in vecs]
        probs = [v.npvalue() for v in vecs]
        tags = []
        for prb in probs:
            tag = np.argmax(prb)
            tags.append(self.vt.i2w[tag])
        return zip(words, tags)


    def evaluate(self, sents, verbose=False):
        good_sent = bad_sent = good = bad = 0.0
        good_sent_relaxed = bad_sent_relaxed = 0.0
        good_relaxed = bad_relaxed = 0.0
        num_words = 0.0

        results = []
        for sent in sents:
            words = [w for w,t in sent]
            golds = [t for w,t in sent]
            tags  = [t for w,t in self.tag_sent(words)]
            results.append([(w,t) for w,t in zip(words,tags)])

            """
            print(golds)
            print(tags)
            raw_input()
            """

            stripped_tags = strip_spines(tags)
            stripped_golds = strip_spines(golds)

            num_words += len(words)
            
            if tags == golds: good_sent += 1
            else: bad_sent += 1
            
            if stripped_tags == stripped_golds: good_sent_relaxed += 1
            else: bad_sent_relaxed += 1

            for go,gu in zip(golds, tags):
                if go == gu: good += 1
                else: bad += 1
                
            for go,gu in zip(stripped_golds, stripped_tags):
                if go == gu: good_relaxed += 1
                else: bad_relaxed += 1

        accuracy = good/(good+bad)
        accuracy_relaxed = good_relaxed/(good_relaxed+bad_relaxed)
        sent_percent = good_sent/(good_sent+bad_sent)
        sent_relaxed_percent = good_sent_relaxed/(good_sent_relaxed+bad_sent_relaxed)

        if verbose:
            print "WORDS EXACT MATCH:  ", accuracy, "\t", 
            print "SENTENCE EXACT MATCH:  ", sent_percent
            print "WORDS RELAXED MATCH:", accuracy_relaxed, "\t",
            print "SENTENCE RELAXED MATCH:", sent_relaxed_percent
        return results, (accuracy, accuracy_relaxed, sent_percent, sent_relaxed_percent)


    def run_train(self, epochs=50):
        best_accuracy = 0
        num_tagged = cum_loss = 0
        num_words = num_spines = 0

        for ITER in xrange(epochs):
            random.shuffle(self.train)
            for i,s in enumerate(self.train,1):
                if i > 0 and i % 500 == 0:   # print status
                    self.trainer.status()
                    print cum_loss / num_tagged
                    cum_loss = num_tagged = 0
                    num_tagged = 0
                if i % 10000 == 0 or i == len(self.train)-1: # eval on dev
                    good_sent = bad_sent = good = bad = 0.0
                    for sent in self.dev:
                        words = [w for w,t in sent]
                        golds = [t for w,t in sent]
                        tags = [t for w,t in self.tag_sent(words)]
                        if tags == golds: good_sent += 1
                        else: bad_sent += 1
                        for go,gu in zip(golds,tags):
                            if go == gu: good += 1
                            else: bad += 1

                    accuracy = good/(good+bad)
                    if accuracy > best_accuracy:
                        print "saving model"
                        self.save_model()
                        best_accuracy = accuracy
                    print accuracy, good_sent/(good_sent+bad_sent)
                # train on sent
                words = [w for w,t in s]
                golds = [t for w,t in s]
                loss_exp =  self.sent_loss(words, golds)
                cum_loss += loss_exp.scalar_value()
                num_tagged += len(golds)
                loss_exp.backward()
                self.trainer.update()
            print "epoch %r finished" % ITER
            self.trainer.update_epoch(1.0)


def get_tagger(train_file, dev_file, model_file):
    train = list(read(train_file))
    dev = list(read(dev_file))
    tagger = Tagger(train, dev, model_file)
    tagger.load_model()
    return tagger


def get_baseline():
    return get_tagger(
        "data/02-21.cleangold.tags",
        "data/22.cleangold.tags",
        "models/baseline",
    )

def get_supertagger():
    return get_tagger(
        "data/02-21.cleangold.supertags",
        "data/22.cleangold.supertags",
        "models/supertags_naive",
    )


def write_output(output_file, results):
    output = open(output_file, "w")
    for result in results:
        string = " ".join([w+"/"+t for w,t in result]) + "\n"
        output.write(string)
    output.close()


def train_tagger(train_file, dev_file, model_file, load_model=False):
    train=list(read(train_file))
    dev=list(read(dev_file))
    tagger = Tagger(train, dev, model_file)
    if load_model:
        tagger.load_model()
    tagger.run_train()
    return tagger
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(prog="Spine-Based Tagger")
    parser.add_argument(
        "--train",
        dest="train",
        help="Training file. (Format is \"word1/tag1 word2/tag2 ...\")",
    )
    parser.add_argument(
        "--dev",
        dest="dev",
        help="Development file.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=None,
        help="Test file.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="Model filename to save to/load from.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Specify output file of evaluated tags (if test file is specified)",
    )
    args = parser.parse_args()




    """
    # format of files: each line is "word1/tag2 word2/tag2 ..."
    train_file, dev_file = args.train, args.dev
    test_file = args.test
    model_file = args.model

    tagger = train_tagger(train_file, dev_file, model_file, load_model=False)
    
    #tagger.run_train()
    #results, stats = tagger.evaluate(dev)



    if args.test is not None:
        test=list(read(test_file))
        results, stats = tagger.evaluate(test)
        if args.output is not None:
            write_output(args.output, results)

    """


        
