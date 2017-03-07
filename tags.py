#!/usr/bin/python
# coding=utf-8
import sys
from pprint import pprint
from features import *
from phrase_tree import *

DELIMITER = "/"

def extractSupertags(tree):
    spines, total = 0,0
    queue = [tree]
    while True:
        terminate = True
        newqueue = []
        for t in queue:
            unary_spine = False
            candidate = t
            while len(candidate.children) == 1:
                candidate = candidate.children[0]
                unary_spine = True

            if unary_spine:
                spines += 1
            total += 1

            if len(candidate.children) == 0:
                newqueue.append(t)
                continue

            terminate = False
            for child in t.children:
                newqueue.append(child)
        queue = newqueue
        if terminate:
            break
    return queue

def processtag(tree):

    """if len(tree.children) == 0:
        word, tag = tree.sentence[tree.leaf]
        return word + DELIMITER + tag
    """

    symbols = ""
    while len(tree.children) == 1:
        symbols = "-" + tree.symbol + symbols
        tree = tree.children[0]
    if symbols == "-":
        symbols = ""
        

    word, tag = tree.sentence[tree.leaf]

    #print(word, DELIMITER, tag, symbols)
    #raw_input()

    return word + DELIMITER + tag + symbols
    



if __name__ == "__main__":
    filename = sys.argv[1]
    output = open(filename + ".supertags", "w")
    fm = FeatureMapper(filename)

    trees = PhraseTree.load_treefile(filename)
    #print trees

    for tree in trees:
        supertags = extractSupertags(tree)
        
        #for tag in supertags:
            #print tag,
            #print processtag(tag)


        supertags = [processtag(tag) for tag in supertags]

        output.write(" ".join(supertags) + "\n")

