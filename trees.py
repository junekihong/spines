#!/usr/bin/python

import sys
from features import *
from phrase_tree import *


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

        t.children = children


    return tree, (spines, total)




if __name__ == "__main__":
    filename = sys.argv[1]
    output = open(filename + ".trunc", "w")
    fm = FeatureMapper(filename)

    trees = PhraseTree.load_treefile(filename)


    spine_words, spine_sents = 0,0
    total_words, total_sents = 0,len(trees)
    for tree in trees:
        #supertags = extractSupertags(tree)
        
        #for tag in supertags:
            #print tag,
            #print processtag(tag)


        #supertags = [processtag(tag) for tag in supertags]

        #output.write(" ".join(supertags) + "\n")

        #print(tree)

        tree, (spines, total_sent) = truncate(tree)
        spine_words += spines
        total_words += total_sent

        if spines > 0:
            spine_sents += 1
            

        tree = PhraseTree("TOP", [tree], tree.sentence)
        #print(tree.pretty())
        #print(truncate(tree)[0])
        #raw_input()

        output.write(str(tree) + "\n")
    print "SPINE WORDS: {}/{} = {}".format(spine_words, total_words, float(spine_words)/total_words)
    print "SPINE SENTS: {}/{} = {}".format(spine_sents, total_sents, float(spine_sents)/total_sents)
