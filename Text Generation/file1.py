# markov_chain_text_generator.py

import random

class MarkovChainTextGenerator:
    def __init__(self, order=1):
        self.order = order
        self.transitions = {}

    def train(self, text):
        words = text.split()
        for i in range(len(words) - self.order):
            key = tuple(words[i:i+self.order])
            next_word = words[i+self.order]
            self.transitions.setdefault(key, []).append(next_word)

    def generate(self, size=50):
        key = random.choice(list(self.transitions.keys()))
        result = list(key)

        for _ in range(size):
            next_words = self.transitions.get(key)
            if not next_words:
                break
            next_word = random.choice(next_words)
            result.append(next_word)
            key = tuple(result[-self.order:])

        return ' '.join(result)


if __name__ == "__main__":
    with open("data/markov_train.txt", "r", encoding='utf-8') as f:
        corpus = f.read()

    markov = MarkovChainTextGenerator(order=2)
    markov.train(corpus)

    generated_text = markov.generate(100)
    print("Generated Text:\n", generated_text)
