from collections import defaultdict, Counter
import math

class MultinomialNaiveBayes:
    def __init__(self):
        self.word_tag_counts = defaultdict(Counter)  # word_tag_counts[tag][word]
        self.tag_counts = Counter()
        self.vocab = set()
        self.total_tags = 0
        self.trained = False

    def train(self, tagged_sentences):
        for sentence in tagged_sentences:
            for word, tag in sentence:
                word = word.lower()
                self.word_tag_counts[tag][word] += 1
                self.tag_counts[tag] += 1
                self.vocab.add(word)

        self.total_tags = sum(self.tag_counts.values())
        self.trained = True

    def predict(self, word):
        word = word.lower()
        best_tag = None
        best_log_prob = float('-inf')
        V = len(self.vocab)

        for tag in self.tag_counts:
            # Prior: P(tag)
            tag_prob = self.tag_counts[tag] / self.total_tags

            # Likelihood: P(word|tag) with Laplace smoothing
            word_given_tag = (self.word_tag_counts[tag][word] + 1) / (self.tag_counts[tag] + V)

            log_prob = math.log(tag_prob) + math.log(word_given_tag)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_tag = tag

        return best_tag

    def tag(self, sentence):
        return [(word, self.predict(word)) for word in sentence]