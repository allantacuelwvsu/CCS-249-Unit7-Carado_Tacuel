# HMM Class Copied from:
# https://github.com/geniusqwerty12/CCS-249---Playground/blob/master/Unit%206/pos_hmm.py

import numpy as np
from collections import defaultdict

class HiddenMarkovModel:
    def __init__(self):
        self.states = set()
        self.vocab = set()
        self.start_probs = defaultdict(float)
        self.trans_probs = defaultdict(lambda: defaultdict(float))
        self.emit_probs = defaultdict(lambda: defaultdict(float))

    def train(self, tagged_sentences):
        tag_counts = defaultdict(int)
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))
        start_counts = defaultdict(int)

        for sentence in tagged_sentences:
            prev_tag = None
            for i, (word, tag) in enumerate(sentence):
                self.states.add(tag)
                self.vocab.add(word)
                tag_counts[tag] += 1
                emission_counts[tag][word] += 1

                if i == 0:
                    start_counts[tag] += 1
                if prev_tag is not None:
                    transition_counts[prev_tag][tag] += 1
                prev_tag = tag

        total_starts = sum(start_counts.values())
        for tag in self.states:
            self.start_probs[tag] = start_counts[tag] / total_starts
            for next_tag in self.states:
                self.trans_probs[tag][next_tag] = (
                    transition_counts[tag][next_tag] / tag_counts[tag]
                )
            for word in self.vocab:
                self.emit_probs[tag][word] = (
                    emission_counts[tag][word] / tag_counts[tag]
                )

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        for tag in self.states:
            V[0][tag] = self.start_probs[tag] * self.emit_probs[tag].get(sentence[0], 1e-6)
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for curr_tag in self.states:
                max_prob, best_prev_tag = max(
                    (V[t - 1][prev_tag] * self.trans_probs[prev_tag].get(curr_tag, 1e-6) *
                     self.emit_probs[curr_tag].get(sentence[t], 1e-6), prev_tag)
                    for prev_tag in self.states
                )
                V[t][curr_tag] = max_prob
                new_path[curr_tag] = path[best_prev_tag] + [curr_tag]

            path = new_path

        max_final_prob, best_final_tag = max((V[-1][tag], tag) for tag in self.states)
        return path[best_final_tag]