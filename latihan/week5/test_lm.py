import numpy as np
from collections import defaultdict

corpus = (
"Artificial Intelligence is amazing. " 
"Artificial Intelligence helps us learn fast. "
"Artificial Intelligence is key to success. "
"Artificial Intelligence systems process data quickly."
"People often overestimate Artificial Intelligence as being universally superior to humans at everything."
)

words = corpus.lower().replace(".","").split()
trigram_counts= defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

vocab = sorted(list(set(words)))

for i in range(len(words)-2):
    w1=words[i]
    w2=words[i+1]
    w3=words[i+2]
    trigram_counts[w1][w2][w3] += 1

k = 0.01
trigram_probabilities = {}

for w1, inner_dict in trigram_counts.items():
    trigram_probabilities[w1]={}
    for w2 , next_word_counts in inner_dict.items():
        total_counts = sum(next_word_counts.values())
        vocab_size = len(vocab)
        trigram_probabilities[w1][w2] = {} 
        for target_word in vocab:
            count = next_word_counts.get(target_word,0)
            prob = (count+k)/(total_counts+k*vocab_size)
            trigram_probabilities[w1][w2][target_word] = prob

print("ini probabilitas trigram: \n", trigram_probabilities)


def generate_trigram_sentence(w1,w2, length=6):
    sentence = [w1,w2]
    
    for _ in range(length):
        if w1 not in trigram_probabilities or w2 not in trigram_probabilities[w1]:
            next_word = np.random.choice(vocab)
        else:
            next_words_dist = trigram_probabilities[w1][w2]
            possible_word
    pass    