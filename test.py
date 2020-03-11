#!/usr/bin/env python3

from hmm import HMM
import os
import re
import subprocess

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
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            
            # Add the encoded word.
            obs_elem.append(obs_map[word])
        
        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map

def obs_map_reverser(obs_map):
    obs_map_r = {}

    for key in obs_map:
        obs_map_r[obs_map[key]] = key

    return obs_map_r

def sample_sentence(hmm, obs_map, n_words=100):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_words)
    sentence = [obs_map_r[i] for i in emission]

    return ' '.join(sentence).capitalize() + '...'

text = open('data/shakespeare.txt').read()
obj, obs_map = parse_observations(text)


n_unique = len(obs_map.keys())
n_train = len(obj)
args = ['./hmm_trainer', '16', '100', str(n_unique), str(n_train)]
for x in obj:
    args.append(str(len(x)))
    for i in x:
        args.append(str(i))

# hmm8 = HMM.unsupervised_HMM(obj, 16, 100)

subprocess.run(args)

print(sample_sentence(hmm8, obs_map, n_words=10))