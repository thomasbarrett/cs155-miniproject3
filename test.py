#!/usr/bin/env python3

from hmm.HMM import HiddenMarkovModel
from hmm.HMM import unsupervised_HMM
import os
import re
import subprocess
import numpy as np
import time

def parse_observations(text):
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []
        
        for word in line:
            word = re.sub(r'[^\w]', '', word).lower()
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
    return ' '.join(sentence).capitalize() + '...'

text = open('data/shakespeare.txt').read()
obj, obs_map = parse_observations(text)

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
hmm = train_unsupervised_hmm(obj, obs_map, 8, 100)
end = time.time()
elapsed1 = end - start
print('elapsed time: {:.4} seconds'.format(end - start))
print(sample_sentence(hmm, obs_map, n_words=10))
print()

start = time.time()
hmm = unsupervised_HMM(obj, 8, 100)
end = time.time()
elapsed2 = end - start
print('elapsed time: {:.4} seconds'.format(end - start))
print(sample_sentence(hmm, obs_map, n_words=10))

print(f'python is {int(elapsed2 / elapsed1)}x slower!')