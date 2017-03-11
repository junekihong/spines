#!/usr/bin/python

import sys
from features import *
from phrase_tree import *
from tagger import *


def truncate(tree):
    spines, total = 0,len(tree.sentence)

    queue = [tree]
    for t in queue:

        children = []
        for child in t.children:
            
            candidate = child
            symbols = ""
            while len(candidate.children) == 1:
                symbols = "-" + candidate.symbol + symbols
                candidate = candidate.children[0]
                
                
            if symbols and candidate.leaf is not None:
                word,tag = tree.sentence[candidate.leaf]
                tag = tag+symbols
                tree.sentence[candidate.leaf] = (word,tag)
                
                new = PhraseTree(tag, [], tree.sentence, candidate.leaf)
                spines += 1
                
                #print "\t", child.symbol, symbols
                #print "\t", child.leaf, tree.sentence[child.leaf]
                #print "\tNEW:", new
                children.append(new)
            else:
                children.append(child)
                queue.append(child)

                #children.append(candidate)
                #queue.append(candidate)


        t.children = children


    return tree, (spines, total)




"""
def propagate_tags(tree, sentence):
    node = tree

    leaves = []

    queue = [node]
    for n in queue:
        if n.leaf is not None:
            leaves.append(n)

        for child in n.children:
            print child,
            queue.append(child)
        raw_input()

    for l in leaves:
        print l, 

    print sentence
    #tree.propagate_sentence(sentence)
"""


if __name__ == "__main__":
    filename = sys.argv[1]
    output = open(filename + ".trunc.auto", "w")
    fm = FeatureMapper(filename)
    trees = PhraseTree.load_treefile(filename)

    tagger = get_supertagger()


    #supertags = open(filename + ".supertags", "w")

    

    spine_words, spine_sents = 0,0
    total_words, total_sents = 0,len(trees)
    for tree in trees:
        tree, (spines, total_sent) = truncate(tree)


        #print tree.sentence
        auto_sent = tagger.tag_sent([w for w,t in tree.sentence])
        #print auto_sent
        #raw_input()
        

        tree.propagate_sentence(auto_sent)

        #raw_input()

        spine_words += spines
        total_words += total_sent
        if spines > 0:
            spine_sents += 1
        tree = PhraseTree("TOP", [tree], tree.sentence)

        """
        string = " ".join([w+"/"+t for w,t in tree.sentence]) + "\n"
        supertags.write(string)
        #print tree.sentence
        #raw_input()
        """

        output.write(str(tree) + "\n")
    print "SPINE WORDS: {}/{} = {}".format(spine_words, total_words, float(spine_words)/total_words)
    print "SPINE SENTS: {}/{} = {}".format(spine_sents, total_sents, float(spine_sents)/total_sents)


