

import sys

from phrase_tree import *
#from features import *
from tagger import *
from trees import *

    
if __name__ == "__main__":
    i = int(sys.argv[1])
    
    train_tags = "data/10way/02-21.cleangold.supertags.split" + str(i)
    dev_tags = "data/10way/02-21.cleangold.supertags.split" + str(i) + ".held"
    model_tags = "data/10way/model_tagger_" + str(i)
    output_tags = "data/10way/output_tags_" + str(i)
    
    tagger = train_tagger(train_tags, dev_tags, model_tags, load_model=False)

    """
    train = list(read(train_tags))
    dev = list(read(dev_tags))
    tagger = Tagger(train, dev, model_tags)
    tagger.load_model()
    """
    
    dev = list(read(dev_tags))
    results, stats = tagger.evaluate(dev)
    write_output(output_tags, results)
    
    

    #train_trees = "data/10way/02-21.cleangold.split" + str(i)
    dev_trees = "data/10way/02-21.cleangold.split" + str(i) + ".held"
    output_trees = "data/10way/output_trees_" + str(i)
    output = open(output_trees, "w")

    #trees_train = PhraseTree.load_treefile(train_file)
    trees = PhraseTree.load_treefile(dev_trees)
    
    for tree in trees:
        tree, (spines, total_sent) = truncate(tree)
        
        auto_sent = tagger.tag_sent([w for w,t in tree.sentence])
        tree.propagate_sentence(auto_sent)
        
        tree = PhraseTree("TOP", [tree], tree.sentence)

        output.write(str(tree) + "\n")
        
    output.close()
