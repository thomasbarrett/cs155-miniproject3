#!/usr/bin/env python3

from hmm.HMM import HiddenMarkovModel
from hmm.HMM import unsupervised_HMM
import os
import re
import subprocess
import numpy as np
import time

def word_pairs(line):
    pairs = []
    words = line.split()
    for i in range(int(len(words)/2)):
        pairs.append(words[2 * i] + ' ' + words[2 * i + 1])
    return pairs

def strip_punctuation(text):
    exclude = [',','.',';',':','!','?','(',')']
    return ''.join(c for c in text if c not in exclude)

def last_word(text):
    return text.split()[-1]

def split_stanzas(text):
    stanzas = []
    stanza = []
    for line in text.split('\n'):
        if line == '' and len(stanza) == 14:
            stanzas.append(stanza)
            stanza = []
        elif len(line.split()) > 1:
            stanza.append(line.strip().lower())
    return stanzas

def gather_rhymes(w, lst, dic):
    for rhyme in dic[w]:
        if not rhyme in lst:
            lst.append(rhyme)
            gather_rhymes(rhyme, lst, dic)

def build_rhyming_dictionary(text, obs_map):
    rhyme_pattern = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 13, 12]
    rhymes_used = {}
    rhymes = {}

    for stanza in split_stanzas(text):
        for a, b in enumerate(rhyme_pattern):
            a_last = strip_punctuation(last_word(stanza[a]))
            b_last = strip_punctuation(last_word(stanza[b]))
            if a_last in rhymes_used:
                rhymes_used[a_last].append(b_last)
            else:
                rhymes_used[a_last] = [b_last]

    for word in rhymes_used.keys():
        all_rhymes = []
        if word in obs_map:
            gather_rhymes(word, all_rhymes, rhymes_used)
            rhymes[obs_map[word]] = [obs_map[rhyme] for rhyme in all_rhymes if rhyme in obs_map and rhyme != word]
    return rhymes

def parse_observations(text):
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n') if line.split()]
    # lines = [word_pairs(line) for line in text.split('\n') if len(word_pairs(line)) > 1]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []
        
        for word in line:
            word = strip_punctuation(word.lower().strip())
            if word not in obs_map:
                obs_map[word] = obs_counter
                obs_counter += 1
            obs_elem.append(obs_map[word])
        obs.append(obs_elem)

    return obs, obs_map

def obs_map_reverser(obs_map):
    obs_map_r = {}
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    return obs_map_r

def sample_sentence(hmm, obs_map, n_words=100):
    obs_map_r = obs_map_reverser(obs_map)
    emission, states = hmm.generate_emission(n_words)
    sentence = [obs_map_r[i] for i in emission]
    return ' '.join(sentence).capitalize()

def generate_stanza(hmm, obs_map, rhymes):
    stanza = []
    rhyme_pattern = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 13, 12]
    chosen_rhymes = []
    for i in range(14):   
        if rhyme_pattern[i] > i:
            obs_map_r = obs_map_reverser(obs_map)
            emission, states = hmm.generate_emission(6)
            sentence = [obs_map_r[i] for i in emission]
            chosen_rhyme = np.random.choice(list(rhymes.keys()))
            chosen_rhymes.append(chosen_rhyme)
            sentence.append(obs_map_r[chosen_rhyme])
            stanza.append(' '.join(sentence).capitalize())
        else:
            obs_map_r = obs_map_reverser(obs_map)
            emission, states = hmm.generate_emission(6)
            sentence = [obs_map_r[i] for i in emission]
            chosen_rhyme = np.random.choice(rhymes[chosen_rhymes[rhyme_pattern[i]]])
            chosen_rhymes.append(chosen_rhyme)
            sentence.append(obs_map_r[chosen_rhyme])
            stanza.append(' '.join(sentence).capitalize())

    return '\n'.join(stanza)

text = open('data/shakespeare.txt').read()
obj, obs_map = parse_observations(text)
rhymes = build_rhyming_dictionary(text, obs_map)

def train_unsupervised_hmm(seqs, obs_map, hidden_states, epochs):
    n_unique = len(obs_map.keys()) + 1
    n_train = len(seqs)
    args = [
        './hmm_trainer', 
        str(hidden_states),
        str(epochs),
        str(n_unique),
        str(n_train)
    ]
    for seq in seqs:
        args.append(str(len(seq)))
        args += list(map(str, seq))

    subprocess.run(args)

    A = np.loadtxt(open("A.csv", "rb"), delimiter=",")
    O = np.loadtxt(open("O.csv", "rb"), delimiter=",")
    return HiddenMarkovModel(A, O)

start = time.time()
hmm = train_unsupervised_hmm(obj, obs_map, 16, 500)
end = time.time()
elapsed1 = end - start
print('elapsed time: {:.4} seconds'.format(end - start))
print()
print(generate_stanza(hmm, obs_map, rhymes))
print()
