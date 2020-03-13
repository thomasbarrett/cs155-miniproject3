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
    lines = [x for sublist in split_stanzas(text) for x in sublist]
    lines = [line.split() for line in lines]
    # lines = [word_pairs(line) for line in text.split('\n') if len(word_pairs(line)) > 1]
    obs_counter = 1
    obs = []
    obs_map = {',': 0}

    for line in lines:
        obs_elem = []
        for word in line:
            word2 = strip_punctuation(word.lower().strip())
            if word2 not in obs_map:
                obs_map[word2] = obs_counter
                obs_counter += 1
            obs_elem.append(obs_map[word2])
            if word.strip()[-1] == ',':
                obs_elem.append(obs_map[','])
        obs.append(obs_elem)

    return obs, obs_map

def build_syllable_dictionary(obs_map):
    syllables_dict = {}
    syllables_dict[','] = {0}
    with open('./data/Syllable_dictionary.txt', newline='') as file:
        for l in file:
            line = l.split()
            if line[0] in obs_map.keys():
                # if it has a different syllable count when at end of line
                # (1st character of 2nd elt of line is an E)
                if line[1][0] == 'E':
                    syllables_dict[obs_map[line[0]]] = (int(line[1][1:]), int(line[2]))

                # though most entries look like 'word E1 2', some (annoyingly) are 'word 2 E1'
                elif line[-1][0] == 'E':
                    syllables_dict[obs_map[line[0]]] = (int(line[-1][1:]), int(line[1]))
                
                else:
                    # TODO: case where word has 2 options for number of syllables
                    syllables_dict[obs_map[line[0]]] = (int(line[-1]), int(line[-1]))

    return syllables_dict

    # syllables = pd.DataFrame.from_records(syllables_list, columns=['word', 'end_syl', 'syl'])

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
    syllables = build_syllable_dictionary(obs_map)
    punctuation = [',','.',';',':','!','?']
    punctuation_weights = [0.35, 0.25, 0.2, 0.05, 0.1, 0.05]

    def count_syllables(emission):
        acc = 0
        for x in emission:
            if x in syllables:
                acc += syllables[x][0]
            elif x != 0: 
                # 0 is comma, which has 0 syllables. Everything else assume 2.
                acc += 2
        return acc
    
    def constraintA(i, last):
        return i in rhymes.keys() if last else True
    
    def constraintB(i, last):
        if last and chosen_rhymes[rhyme_pattern[line_index]] in rhymes:
            return i in rhymes[chosen_rhymes[rhyme_pattern[line_index]]]
        else:
            return True

    def capitalize_proper_nouns(sentence1):
        sentence2 = []
        for word in sentence1:
            if word == 'i':
                sentence2.append('I')
            else:
                sentence2.append(word)
        return sentence2

    def capitalize_first(line):
        line = line[0].capitalize() + line[1:]
        return line

    for line_index in range(14):   
        obs_map_r = obs_map_reverser(obs_map)
        if rhyme_pattern[line_index] > line_index:
            emission, states = hmm.generate_emission(constraintA, count_syllables, 10)
        else:
            emission, states = hmm.generate_emission(constraintB, count_syllables, 10)
        chosen_rhymes.append(emission[-1])
        line = capitalize_first(' '.join(capitalize_proper_nouns([obs_map_r[i] for i in emission])))
        stanza.append(''.join([c for (i, c) in enumerate(line) if not (c == ' ' and line[i + 1] == ',')]))

    stanza = list(map(lambda l: l + np.random.choice(punctuation, p=punctuation_weights), stanza))
    return '\n'.join(stanza[:-2]) + '\n  ' + stanza[-2] + '\n  ' + stanza[-1] 

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

def load_hmm():
    A = np.loadtxt(open("A.csv", "rb"), delimiter=",")
    O = np.loadtxt(open("O.csv", "rb"), delimiter=",")
    return HiddenMarkovModel(A, O)

start = time.time()
hmm = train_unsupervised_hmm(obj, obs_map, 20, 500)
end = time.time()
elapsed1 = end - start
print('elapsed time: {:.4} seconds'.format(end - start))
print()

file = open('generated-poem.txt','w+') 
for stanza in range(20):
    file.write(str(stanza + 1).center(40, ' ') + '\n')
    file.write(generate_stanza(hmm, obs_map, rhymes))
    file.write('\n\n')
file.close()  


