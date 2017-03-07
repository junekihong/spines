from collections import Counter, defaultdict
from itertools import count
import random
import dynet as dy
import numpy as np

#Taken from https://github.com/clab/dynet_tutorial_examples/blob/master/tutorial_bilstm_tagger.py

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

def word_rep(w, cf_init, cb_init):
    if wc[w] > 5:
        w_index = vw.w2i[w]
        return WORDS_LOOKUP[w_index]
    else:
        pad_char = vc.w2i["<*>"]
        char_ids = [pad_char] + [vc.w2i[c] for c in w] + [pad_char]
        char_embs = [CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

def build_tagging_graph(words):
    dy.renew_cg()
    # parameters -> expressions
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    # initialize the RNNs
    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()

    cf_init = cFwdRNN.initial_state()
    cb_init = cBwdRNN.initial_state()

    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    wembs = [word_rep(w, cf_init, cb_init) for w in words]
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

def sent_loss(words, tags):
    vecs = build_tagging_graph(words)
    errs = []
    for v,t in zip(vecs,tags):
        tid = vt.w2i[t]
        err = dy.pickneglogsoftmax(v, tid)
        errs.append(err)
    return dy.esum(errs)

def tag_sent(words):
    vecs = build_tagging_graph(words)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in probs:
        tag = np.argmax(prb)
        tags.append(vt.i2w[tag])
    return zip(words, tags)


def strip_spines(tags):
    result = []
    for t in tags:
        if "-" in t and (t != "-LRB-" and t != "-RRB-"):
            t = t.split("-")[0]
        result.append(t)
    return result



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
    args = parser.parse_args()

        

    # format of files: each line is "word1/tag2 word2/tag2 ..."
    train_file, dev_file = args.train, args.dev
    test_file = args.test
    model_file = args.model


    train=list(read(train_file))
    dev=list(read(dev_file))
    test = None
    if test_file is not None:
        test=list(read(test_file))


    words,tags,chars = [],[],set()
    wc=Counter()
    for sent in train:
        for w,p in sent:
            words.append(w)
            tags.append(p)
            chars.update(w)
            wc[w]+=1
    words.append("_UNK_")
    chars.add("<*>")
    
    vw = Vocab.from_corpus([words]) 
    vt = Vocab.from_corpus([tags])
    vc = Vocab.from_corpus([chars])
    UNK = vw.w2i["_UNK_"]
    
    nwords = vw.size()
    ntags  = vt.size()
    nchars  = vc.size()
    
    # DyNet Starts
    
    model = dy.Model()
    trainer = dy.AdamTrainer(model)
    
    WORDS_LOOKUP = model.add_lookup_parameters((nwords, 128))
    CHARS_LOOKUP = model.add_lookup_parameters((nchars, 20))
    p_t1  = model.add_lookup_parameters((ntags, 30))
    
    # MLP on top of biLSTM outputs 100 -> 32 -> ntags
    pH = model.add_parameters((32, 50*2))
    pO = model.add_parameters((ntags, 32))
    
    # word-level LSTMs
    fwdRNN = dy.LSTMBuilder(1, 128, 50, model) # layers, in-dim, out-dim, model
    bwdRNN = dy.LSTMBuilder(1, 128, 50, model)
    
    # char-level LSTMs
    cFwdRNN = dy.LSTMBuilder(1, 20, 64, model)
    cBwdRNN = dy.LSTMBuilder(1, 20, 64, model)
    
    



    # Main
    best_accuracy = 0
    num_tagged = cum_loss = 0

    num_words = num_spines = 0

    if test is not None:
        model.load(model_file)
        good_sent = bad_sent = good = bad = 0.0
        good_sent_relaxed = bad_sent_relaxed = 0.0
        good_relaxed = bad_relaxed = 0.0

        for sent in test:
            words = [w for w,t in sent]
            golds = [t for w,t in sent]
            tags  = [t for w,t in tag_sent(words)]

            stripped_tags = strip_spines(tags)
            stripped_golds = strip_spines(golds)

            num_words += len(words)

            """
            for t in golds:
                print t,
            print
            for t in tags:
                print t,
            print
            print
            raw_input()
            """

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


        print "WORDS EXACT MATCH:  ", accuracy, "\t", 
        print "SENTENCE EXACT MATCH:  ", good_sent/(good_sent+bad_sent)
        print "WORDS RELAXED MATCH:", accuracy_relaxed, "\t",
        print "SENTENCE RELAXED MATCH:", good_sent_relaxed/(good_sent_relaxed+bad_sent_relaxed)
        exit()

    for ITER in xrange(50):
        random.shuffle(train)
        for i,s in enumerate(train,1):
            if i > 0 and i % 500 == 0:   # print status
                trainer.status()
                print cum_loss / num_tagged
                cum_loss = num_tagged = 0
                num_tagged = 0
            if i % 10000 == 0 or i == len(train)-1: # eval on dev
                good_sent = bad_sent = good = bad = 0.0
                for sent in dev:
                    words = [w for w,t in sent]
                    golds = [t for w,t in sent]
                    tags = [t for w,t in tag_sent(words)]
                    if tags == golds: good_sent += 1
                    else: bad_sent += 1
                    for go,gu in zip(golds,tags):
                        if go == gu: good += 1
                        else: bad += 1

                accuracy = good/(good+bad)
                if accuracy > best_accuracy:
                    print "saving model"
                    model.save(model_file)
                    best_accuracy = accuracy
                print accuracy, good_sent/(good_sent+bad_sent)
            # train on sent
            words = [w for w,t in s]
            golds = [t for w,t in s]
            loss_exp =  sent_loss(words, golds)
            cum_loss += loss_exp.scalar_value()
            num_tagged += len(golds)
            loss_exp.backward()
            trainer.update()
        print "epoch %r finished" % ITER
        trainer.update_epoch(1.0)


