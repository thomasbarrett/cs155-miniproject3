
def word_pairs(line):
    pairs = []
    words = line.split()
    for i in range(int(len(words)/2)):
        pairs.append(words[2 * i] + ' ' + words[2 * i + 1])
    return pairs