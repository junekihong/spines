

# Code taken from James Cross' Span Parser: https://github.com/jhcross/span-parser

from __future__ import print_function
from __future__ import division

import heapq
import numpy as np
from collections import defaultdict
#from cpython cimport bool

from phrase_tree import PhraseTree, FScore
#from beam cimport beamsearch

"""
Shift-Combine-Label parser.
"""
class Parser(object):
  
    def __init__(self, n):
        """
        Initial state for parsing an n-word sentence.
        """
        self.n = n
        self.i = 0
        self.stack = Stack()
        self.step = 0
        self.action_taken = "none"
        self.score = 0
        self.score_delta = 0
        self.previous = None
        self.features = None
        
        
    def __str__(self):
        current = "(i:{}, stacksize:{}, N:{}) ".format(self.n, self.i, len(self.stack))
        stats = "STEP:{}, PREV ACT:{}".format(self.step, self.action_taken)
        stats += ", SCORE:{:<.3f}".format(self.score)
        if self.finished():
            stats += ", (FINISHED)"
        
        return current + stats

            
        
    def copy(self):
        result = Parser(self.n)
        result.i = self.i
        result.stack = self.stack.copy()
        result.step = self.step
        result.action_taken = self.action_taken
        result.score = self.score
        result.score_delta = self.score_delta
        result.previous = self.previous
        result.features = self.features
        return result
        
    def can_shift(self):
        return (self.i < self.n)


    def can_combine(self):
        return (len(self.stack) > 1)


    def shift(self):
        j = self.i  # (index of shifted word)
        treelet = PhraseTree(leaf=j)
        self.stack.push((j, j, [treelet]))
        #self.stack.append((j, j, [treelet]))
        self.i += 1

        
    def combine(self):
        (_, right, treelist0) = self.stack.pop()
        (left, _, treelist1) = self.stack.pop()
        self.stack.push((left, right, treelist1 + treelist0))
        #self.stack.append((left, right, treelist1 + treelist0))


    def label(self, nonterminals=[]):
        for nt in nonterminals:
            (left, right, trees) = self.stack.pop()
            tree = PhraseTree(symbol=nt, children=trees)
            self.stack.push((left, right, [tree]))
            #self.stack.append((left, right, [tree]))


    def take_action(self, action):
        self.previous = self.copy()
        self.features = None
        #self.score = 0
        self.step += 1
        self.action_taken = action
        if action == 'sh':
            self.shift()
        elif action == 'comb':
            self.combine()
        elif action == 'none':
            return
        elif action.startswith('label-'):
            self.label(action[6:].split('-'))
        else:
            raise RuntimeError('Invalid Action: {}'.format(action))


    def finished(self):
        return (
            (self.i == self.n) and 
            (len(self.stack) == 1) and 
            (len(self.stack[0][2]) == 1)
        )


    def tree(self):
        if not self.finished():
            raise RuntimeError('Not finished.')
        return self.stack[0][2][0]


    def s_features(self):
        """
        Features for predicting structural action (shift, combine):
            (pre-s1-span, s1-span, s0-span, post-s0-span)
        Note features use 1-based indexing:
            ... a span of (1, 1) means the first word of sentence
            ... (x, x-1) means no span
        """
        lefts = []
        rights = []

        # pre-s1-span
        lefts.append(1)
        if len(self.stack) < 2:
            rights.append(0)
        else:
            s1_left = self.stack[-2][0] + 1
            rights.append(s1_left - 1)

        # s1-span
        if len(self.stack) < 2:
            lefts.append(1)
            rights.append(0)
        else:
            s1_left = self.stack[-2][0] + 1
            lefts.append(s1_left)
            s1_right = self.stack[-2][1] + 1
            rights.append(s1_right)

        # s0-span
        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1] + 1
            rights.append(s0_right)

        # post-s0-span
        lefts.append(self.i + 1)
        rights.append(self.n)

        return tuple(lefts), tuple(rights)



    def l_features(self):
        """
        Features for predicting label action:
            (pre-s0-span, s0-span, post-s0-span)
        """
        lefts = []
        rights = []

        # pre-s0-span
        lefts.append(1)
        if len(self.stack) < 1:
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            rights.append(s0_left - 1)


        # s0-span
        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1] + 1
            rights.append(s0_right)

        # post-s0-span
        lefts.append(self.i + 1)
        rights.append(self.n)

        return tuple(lefts), tuple(rights)


    def s_oracle(self, tree):
        """
        Returns correct structural action in current (arbitrary) state, 
            given gold tree.
            Deterministic (prefer combine).
        """
        if not self.can_shift():
            return 'comb'
        elif not self.can_combine():
            return 'sh'
        else:
            (left0, right0, _) = self.stack[-1]
            a, _ = tree.enclosing(left0, right0)
            if a == left0:
                return 'sh'
            else:
                return 'comb'


    def l_oracle(self, tree):            
        (left0, right0, _) = self.stack[-1]
        labels = tree.span_labels(left0, right0)[::-1]
        if len(labels) == 0:
            return 'none'
        else:
            return 'label-' + '-'.join(labels)


    @staticmethod
    def gold_actions(tree):
        n = len(tree.sentence)
        state = Parser(n)
        result = []

        for step in range(2 * n - 1):
            if state.can_combine():                
                (left0, right0, _) = state.stack[-1]
                (left1, _, _) = state.stack[-2]
                a, b = tree.enclosing(left0, right0)
                if left1 >= a:
                    result.append('comb')
                    state.combine()
                else:
                    result.append('sh')
                    state.shift()
            else:
                result.append('sh')
                state.shift()

            (left0, right0, _) = state.stack[-1]
            labels = tree.span_labels(left0, right0)[::-1]
            if len(labels) == 0:
                result.append('none')
            else:
                result.append('label-' + '-'.join(labels))
                state.label(labels)

        return result


    @staticmethod
    def training_data(tree):
        """
        Using oracle (for gold sequence), omitting mandatory S-actions
        """
        s_features = []
        l_features = []

        n = len(tree.sentence)
        state = Parser(n)
        result = []

        for step in range(2 * n - 1):

            if not state.can_combine():
                action = 'sh'
            elif not state.can_shift():
                action = 'comb'
            else:
                action = state.s_oracle(tree)
                features = state.s_features()
                s_features.append((features, action))
            state.take_action(action)


            action = state.l_oracle(tree)
            features = state.l_features()
            l_features.append((features, action))
            state.take_action(action)

        return (s_features, l_features)


    # MOVED TO: ml.pyx
    """
    @staticmethod
    def parse_ml(sentence, fm, args, ml):
        cdef int n
        cdef Parser state, top_state
        cdef float top_score
        cdef dict beam
        n = len(sentence)
        state = Parser(n)
        w,t = fm.sentence_sequences(sentence)

        beam = beamsearch(sentence, fm, ml, args, None)
        top_of_beam = heapq.nlargest(1, beam[len(beam)-1])[0]
        top_score,top_state,_ = top_of_beam
        #if not top_state.finished():
        #    raise RuntimeError('Bad ending state!')
        tree = top_state.stack[0][2][0]
        tree.propagate_sentence(sentence)
        return tree, top_score
        """

    """
    @staticmethod
    def evaluate_corpus_ml(trees, fm, args, ml):
        best_score = 0
        accuracy = FScore()
        for tree in trees:
            predicted, score = Parser.parse_ml(tree.sentence, fm, args, ml)
            local_accuracy = predicted.compare(tree)
            accuracy += local_accuracy
            best_score = max(best_score, score)
        return accuracy, best_score
    """

    
    @staticmethod
    def write_predicted(fname, test_trees, fm, network):
        """
        Input trees being used only to carry sentences.
        """
        f = open(fname, 'w')
        for tree in test_trees:
            predicted = Parser.parse(tree.sentence, fm, network)
            topped = PhraseTree(
                symbol='TOP',
                children=[predicted],
                sentence=predicted.sentence,
            )
            f.write(str(topped))
            f.write('\n')
        f.close()



class Stack:
    """ A Tree-Structured Stack implementation.
    """
    def __init__(self):
        self.stacksize = 0
        self.tail = ()

    def copy(self):
        result = Stack()
        result.stacksize = self.stacksize
        result.tail = self.tail
        return result
        
    def push(self, item):
        self.tail = (item, self.tail)
        self.stacksize += 1

    def pop(self):
        if self.stacksize == 0:
            return
        item = self.tail[0]
        self.tail = self.tail[1]
        self.stacksize -= 1
        return item
        
    def head(self):
        if self.stacksize == 0:
            return None
        return self.tail[0]

    def tail(self):
        return self.tail

    def access(self, index):
        assert self.stacksize > 0
        if self.stacksize == 1:
            index = 0
        else:
            if index < 0:
                index = index % self.stacksize
            index = self.stacksize - index - 1
        iterate = self.tail
        for _ in xrange(index):
            iterate = iterate[1]
        return iterate[0]
    
    def exists(self, item):
        iterate = self.tail
        while iterate:
            if item == iterate[0]:
                return True
            iterate = iterate[1]
        return False
    
    def elements(self):
        iterate = self.tail
        result = []
        while iterate:
            result.insert(0,iterate[0])
            iterate = iterate[1]
        return result
    
    def length(self):
        return self.stacksize

    def __len__(self):
        return self.stacksize

    def __str__(self):
        return "stacksize: {}, ".format(self.stacksize) + str(self.elements())

    def __getitem__(self, key):
        return self.access(key)
