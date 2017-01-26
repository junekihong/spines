#!/usr/bin/python
# coding=utf-8
import sys
from pprint import pprint
import pyparsing as pp
from features import *
from phrase_tree import *

DELIMITER = "/"

def extractSupertags(tree):
    queue = [tree]
    while True:
        terminate = True
        newqueue = []
        for t in queue:
            if len(t.children) == 0:
                newqueue.append(t)
                continue
            if len(t.children) == 1 and len(t.children[0].children) == 0:
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
    if len(tree.children) == 0:
        word, tag = tree.sentence[tree.leaf]
        return word + DELIMITER + tag
    word, tag = tree.sentence[tree.children[0].leaf]
    return word + DELIMITER + tag + "-" + tree.symbol
    



if __name__ == "__main__":
    filename = sys.argv[1]
    output = open(filename + ".supertags", "w")
    fm = FeatureMapper(filename)

    trees = PhraseTree.load_treefile(filename)
    #print trees

    for tree in trees:
        supertags = extractSupertags(tree)
        supertags = [processtag(tag) for tag in supertags]

        output.write(" ".join(supertags) + "\n")

